import argparse
import os
from collections.abc import Callable

import torch
from peft import LoraConfig, get_peft_model

from swift import get_model_processor, get_template, load_dataset
from swift.dataset import LazyLLMDataset
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.utils import (
    find_all_linears,
    get_logger,
    seed_everything,
)

logger = get_logger()


def train_common(
    train_jsonl: str,
    speaker_name: str,
    init_model_path: str,
    output_model_path: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    lr: float,
    min_lr_factor: float,
    warmup_ratio: float,
    seed: int,
    *,
    prepare_model: Callable,
    finalize_model: Callable,
) -> str:
    """Shared pipeline: env → model → prepare → dataset → train → save."""
    # --- environment ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["SPEAKER_NAME"] = speaker_name
    seed_everything(seed)

    # --- hard-coded training configuration ---
    max_length = 4096
    split_dataset_ratio = 0.0
    load_from_cache_file = True
    dataset_num_proc = 1
    output_model_path = os.path.abspath(os.path.expanduser(output_model_path))

    # --- model + processor + template ---
    model, processor = get_model_processor(
        init_model_path, torch_dtype=torch.bfloat16, attn_impl="flash_attention_2"
    )
    template = get_template(processor, max_length=max_length)
    template.set_mode("train")

    # --- model preparation (LoRA or full) ---
    model = prepare_model(model)

    # --- dataset ---
    train_dataset, _ = load_dataset(
        [train_jsonl],
        split_dataset_ratio=split_dataset_ratio,
        num_proc=dataset_num_proc,
        load_from_cache_file=load_from_cache_file,
        seed=seed,
    )
    train_dataset = LazyLLMDataset(
        train_dataset, template.encode, strict=False, random_state=seed
    )
    template.print_inputs(train_dataset[0])

    # --- trainer ---
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_path,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": lr * min_lr_factor},
        warmup_ratio=warmup_ratio,
        optim="adamw_torch_fused",
        bf16=True,
        report_to=["tensorboard"],
        weight_decay=0.01,
        logging_first_step=True,
        save_strategy="no",
        save_only_model=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        use_liger_kernel=True,
        dataloader_num_workers=4,
        seed=seed,
        data_seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        template=template,
        train_dataset=train_dataset,
    )
    trainer.train()

    # --- save (finalize merges LoRA into base if needed) ---
    model = finalize_model(model)
    template.save_callback(model, output_model_path)
    logger.info(f"saved model to: {output_model_path}")
    return output_model_path


def prepare_full(model):
    model.train()
    model.requires_grad_(True)
    model.enable_input_require_grads()  # compatible with gradient checkpointing
    return model


def train_full(
    train_jsonl: str,
    speaker_name: str,
    init_model_path: str,
    output_model_path: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    lr: float,
    min_lr_factor: float,
    warmup_ratio: float,
    seed: int,
) -> str | None:
    return train_common(
        train_jsonl=train_jsonl,
        speaker_name=speaker_name,
        init_model_path=init_model_path,
        output_model_path=output_model_path,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        lr=lr,
        min_lr_factor=min_lr_factor,
        warmup_ratio=warmup_ratio,
        seed=seed,
        prepare_model=prepare_full,
        finalize_model=lambda m: m,
    )


def train_lora(
    train_jsonl: str,
    speaker_name: str,
    init_model_path: str,
    output_model_path: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    lr: float,
    min_lr_factor: float,
    warmup_ratio: float,
    lora_rank: int,
    lora_alpha: int,
    seed: int,
) -> str | None:
    def prepare_lora(model):
        target_modules = find_all_linears(model)
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()  # compatible with gradient checkpointing
        return model

    return train_common(
        train_jsonl=train_jsonl,
        speaker_name=speaker_name,
        init_model_path=init_model_path,
        output_model_path=output_model_path,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        lr=lr,
        min_lr_factor=min_lr_factor,
        warmup_ratio=warmup_ratio,
        seed=seed,
        prepare_model=prepare_lora,
        finalize_model=lambda m: m.merge_and_unload(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Programmatic equivalent of `swift sft` for Qwen3-TTS fine-tuning."
    )
    parser.add_argument("--tuner-type", choices=["lora", "full"], default="full")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--speaker-name", default="cyrene")
    parser.add_argument("--init-model-path", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument(
        "--output-model-path", default="/root/autodl-tmp/output/tts/test"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.29e-5)
    parser.add_argument("--min-lr-factor", type=float, default=0.434)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.tuner_type == "lora":
        last_checkpoint = train_lora(
            train_jsonl=args.train_jsonl,
            speaker_name=args.speaker_name,
            init_model_path=args.init_model_path,
            output_model_path=args.output_model_path,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            lr=args.lr,
            min_lr_factor=args.min_lr_factor,
            warmup_ratio=args.warmup_ratio,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            seed=args.seed,
        )
    elif args.tuner_type == "full":
        last_checkpoint = train_full(
            train_jsonl=args.train_jsonl,
            speaker_name=args.speaker_name,
            init_model_path=args.init_model_path,
            output_model_path=args.output_model_path,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            lr=args.lr,
            min_lr_factor=args.min_lr_factor,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
        )
    else:
        raise Exception(args.tuner_type)

    print(last_checkpoint)


if __name__ == "__main__":
    main()
