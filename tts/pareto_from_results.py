import argparse
import json
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState


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


def build_study(results_path: Path, tuner_type: str) -> optuna.Study:
    search_space = _build_search_space(tuner_type)
    study = optuna.create_study(
        directions=["minimize", "maximize", "minimize"],
        study_name="qwen3-tts-mobo",
    )

    with open(results_path, "r", encoding="utf-8") as f:
        records: list[dict[str, Any]] = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["trial"])

    for r in records:
        params = {key: r[key] for key in search_space}
        values = (r["wer"], r["cosine_sim"], r["utmos_mse"])
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=search_space,
                values=values,
                state=TrialState.COMPLETE,
            )
        )
    return study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--tuner-type",
        choices=["lora", "full"],
        required=True,
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    study = build_study(results_path, args.tuner_type)

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


if __name__ == "__main__":
    main()
