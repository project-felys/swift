import argparse
import gc
import json
from pathlib import Path
from typing import Any

import optuna
import torch
from optuna.trial import TrialState

from eval_nn import EvalNN, evaluate_model
from gen_audio import generate_all, prepare_eval_data
from trainer import train_full, train_lora


def _build_search_space(
    tuner_type: str,
) -> dict[str, optuna.distributions.BaseDistribution]:
    match tuner_type:
        case "full":
            return {
                "lr": optuna.distributions.FloatDistribution(5e-6, 5e-5, log=True),
                "min_lr_factor": optuna.distributions.FloatDistribution(0.1, 1.0),
                "batch_size": optuna.distributions.CategoricalDistribution([4, 8, 16]),
                "num_epochs": optuna.distributions.IntDistribution(2, 6),
            }
        case "lora":
            return {
                "lr": optuna.distributions.FloatDistribution(5e-5, 5e-4, log=True),
                "min_lr_factor": optuna.distributions.FloatDistribution(0.1, 1.0),
                "batch_size": optuna.distributions.CategoricalDistribution([4, 8, 16]),
                "num_epochs": optuna.distributions.IntDistribution(2, 6),
                "lora_rank": optuna.distributions.CategoricalDistribution([4, 8, 16]),
                "lora_alpha": optuna.distributions.CategoricalDistribution([8, 16, 32]),
            }
        case _:
            raise ValueError(f"Unknown tuner_type: {tuner_type!r}")


def _sample_params(trial: optuna.Trial, tuner_type: str) -> dict[str, Any]:
    match tuner_type:
        case "full":
            return {
                "lr": trial.suggest_float("lr", 5e-6, 5e-5, log=True),
                "min_lr_factor": trial.suggest_float("min_lr_factor", 0.1, 1.0),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
                "num_epochs": trial.suggest_int("num_epochs", 2, 6),
            }
        case "lora":
            return {
                "lr": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
                "min_lr_factor": trial.suggest_float("min_lr_factor", 0.1, 1.0),
                "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
                "num_epochs": trial.suggest_int("num_epochs", 2, 6),
                "lora_rank": trial.suggest_categorical("lora_rank", [4, 8, 16]),
                "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32]),
            }
        case _:
            raise ValueError(f"Unknown tuner_type: {tuner_type!r}")


def _run_trial_training(
    tuner_type: str,
    params: dict[str, Any],
    train_jsonl: str,
    speaker_name: str,
    init_model_path: str,
    output_model_path: str,
    seed: int,
) -> str:
    match tuner_type:
        case "full":
            return train_full(
                train_jsonl=train_jsonl,
                speaker_name=speaker_name,
                init_model_path=init_model_path,
                output_model_path=output_model_path,
                batch_size=params["batch_size"] // 2,
                gradient_accumulation_steps=2,
                num_epochs=params["num_epochs"],
                lr=params["lr"],
                min_lr_factor=params["min_lr_factor"],
                warmup_ratio=0.05,
                seed=seed,
            )
        case "lora":
            return train_lora(
                train_jsonl=train_jsonl,
                speaker_name=speaker_name,
                init_model_path=init_model_path,
                output_model_path=output_model_path,
                batch_size=params["batch_size"],
                gradient_accumulation_steps=1,
                num_epochs=params["num_epochs"],
                lr=params["lr"],
                min_lr_factor=params["min_lr_factor"],
                warmup_ratio=0.05,
                lora_rank=params["lora_rank"],
                lora_alpha=params["lora_alpha"],
                seed=seed,
            )
        case _:
            raise ValueError(f"Unknown tuner_type: {tuner_type!r}")


def _create_study(
    results_path: Path,
    search_space: dict[str, optuna.distributions.BaseDistribution],
    n_trials: int,
    seed: int,
) -> tuple[optuna.Study, int]:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=25)
    study = optuna.create_study(
        directions=["minimize", "maximize", "minimize"],
        sampler=sampler,
        study_name="qwen3-tts-mobo",
    )

    completed_records: list[dict[str, Any]] = []
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            all_records: list[dict[str, Any]] = [json.loads(line) for line in f]
        all_records.sort(key=lambda r: r["trial"])

        for record in all_records:
            params = {key: record[key] for key in search_space}
            values = (record["wer"], record["cosine_sim"], record["utmos_mse"])
            frozen_trial = optuna.trial.create_trial(
                params=params,
                distributions=search_space,
                values=values,
                state=TrialState.COMPLETE,
            )
            study.add_trial(frozen_trial)
            completed_records.append(record)

    n_completed = len(completed_records)
    n_remaining = max(0, n_trials - n_completed)

    if n_remaining == 0:
        print(f"All {n_trials} trials already completed. Nothing to do.")
    elif n_completed > 0:
        print(
            f"Resuming from {n_completed} completed trial(s), {n_remaining} remaining."
        )

    return study, n_remaining


def _save_pareto_front(study: optuna.Study, results_path: Path) -> None:
    pareto_front = study.best_trials
    print(f"\nPareto front ({len(pareto_front)} non-dominated solutions):")

    pareto_path = results_path.parent / "pareto_front.jsonl"
    with open(pareto_path, "w", encoding="utf-8") as pf:
        for t in pareto_front:
            wer, cos_sim, utmos_mse = t.values
            line = json.dumps(
                {
                    "trial": t.number,
                    "wer": wer,
                    "cosine_sim": cos_sim,
                    "utmos_mse": utmos_mse,
                    "params": t.params,
                },
                ensure_ascii=False,
            )
            print(line, file=pf)
    print(f"Pareto front saved to {pareto_path}")


def bo_loop(args: argparse.Namespace) -> None:
    output_root = Path(args.output_model_path)
    results_path = output_root / "bo_results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    eval_data: list[dict[str, str]] = prepare_eval_data(args.test_jsonl)
    ref_audio_paths = [Path(item["audio_path"]) for item in eval_data]
    evaluator = EvalNN(ref_audio_paths, language="Chinese")

    search_space = _build_search_space(args.tuner_type)
    study, n_remaining = _create_study(
        results_path, search_space, args.n_trials, args.seed
    )

    with open(results_path, "a", encoding="utf-8") as f:
        for _ in range(n_remaining):
            trial = study.ask(fixed_distributions=search_space)
            params = _sample_params(trial, args.tuner_type)

            trial_output_path = str(output_root / str(trial.number))
            last_ckpt_dir = _run_trial_training(
                tuner_type=args.tuner_type,
                params=params,
                train_jsonl=args.train_jsonl,
                speaker_name=args.speaker_name,
                init_model_path=args.init_model_path,
                output_model_path=trial_output_path,
                seed=args.seed,
            )

            gc.collect()
            torch.cuda.empty_cache()

            saved_dir, stems = generate_all(last_ckpt_dir, eval_data, args.speaker_name)

            gc.collect()
            torch.cuda.empty_cache()

            metrics_mean = evaluate_model(saved_dir, stems, evaluator)
            values = (
                metrics_mean["wer"],
                metrics_mean["cosine_sim"],
                metrics_mean["utmos_mse"],
            )

            study.tell(trial, values)

            gc.collect()
            torch.cuda.empty_cache()

            record = {
                "trial": trial.number,
                "ckpt_dir": last_ckpt_dir,
                **params,
                **metrics_mean.to_dict(),
            }
            print(json.dumps(record, ensure_ascii=False), file=f)
            f.flush()

    _save_pareto_front(study, results_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-objective Bayesian hyperparameter search for Qwen3-TTS fine-tuning."
    )
    parser.add_argument(
        "--tuner-type",
        choices=["lora", "full"],
        help="Fine-tuning mode to search over.",
        required=True,
    )
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--test-jsonl", type=str, required=True)
    parser.add_argument("--speaker-name", type=str, required=True)
    parser.add_argument(
        "--init-model-path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    )
    parser.add_argument("--output-model-path", type=str, default="output")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bo_loop(args)


if __name__ == "__main__":
    main()
