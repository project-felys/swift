CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
nohup swift pt \
    --torch_dtype 'bfloat16' \
    --model 'Qwen/Qwen3.5-4B-Base' \
    --model_type 'qwen3_5' \
    --template 'qwen3_5' \
    --dataset \
        '/root/autodl-tmp/corpora/pt/amphoreus' \
        '/root/autodl-tmp/corpora/pt/amphoreus' \
        '/root/autodl-tmp/corpora/pt/amphoreus' \
        '/root/autodl-tmp/corpora/pt/amphoreus/chs.jsonl' \
        '/root/autodl-tmp/corpora/pt/amphoreus/en.jsonl' \
        '/root/autodl-tmp/corpora/pt/vendor/miyoushe.jsonl' \
        '/root/autodl-tmp/corpora/pt/vendor/miyoushe.jsonl' \
        '/root/autodl-tmp/corpora/pt/vendor/miyoushe.jsonl' \
        '/root/autodl-tmp/corpora/pt/vendor/miyoushe.jsonl' \
        '/root/autodl-tmp/corpora/pt/vendor/miyoushe.jsonl' \
    --split_dataset_ratio '0.0' \
    --max_length '5120' \
    --tuner_type 'full' \
    --learning_rate '2e-5' \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 2e-6}' \
    --num_train_epochs '1.0' \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --eval_steps '50' \
    --save_steps 400 \
    --output_dir '/root/autodl-tmp/pt' \
    --attn_impl 'flash_attention_2' \
    --neftune_noise_alpha 0 \
    --truncation_strategy 'right' \
    --warmup_ratio '0.10' \
    --report_to 'tensorboard' \
    --use_liger_kernel 'True' \
    --dataset_num_proc 8 \
    --dataloader_num_workers 8 \
    --packing \
    --save_only_model \
    > "$(date +%s).log" 2>&1 &
