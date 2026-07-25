from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

TARGET_SR = 24000

BASELINE_MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def prepare_eval_data(test_jsonl: str) -> list[dict[str, str]]:
    lines = [json.loads(ln) for ln in open(test_jsonl, encoding="utf-8") if ln.strip()]

    data: list[dict[str, str]] = []
    for entry in lines:
        data.append(
            {
                "text": entry["messages"][-1]["content"],
                "audio_path": entry["audios"][0],
            }
        )
    return data


def generate_all(
    model_path: str,
    eval_data: list[dict[str, str]],
    speaker_name: str,
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> tuple[str, list[str]]:
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    test_dir = Path(model_path) / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    saved_dir = str(test_dir)

    results: list[str] = []
    pbar = tqdm(total=len(eval_data), desc="Sampling")
    for i in range(0, len(eval_data), batch_size):
        sub = eval_data[i : i + batch_size]
        texts = [e["text"] for e in sub]
        gen_wavs_list, _ = model.generate_custom_voice(
            text=texts,
            speaker=speaker_name,
            max_new_tokens=max_new_tokens,
        )
        for entry, wav in zip(sub, gen_wavs_list):
            if wav.ndim > 1:
                wav = wav.mean(axis=0)
            stem = Path(entry["audio_path"]).stem
            saved_path = str(test_dir / f"{stem}.wav")
            sf.write(saved_path, wav, TARGET_SR)
            results.append(stem)
        pbar.update(len(sub))
    pbar.close()
    return saved_dir, results


def generate_all_baseline(
    eval_data: list[dict[str, str]],
    output_path: Path,
    ref_audio: Path,
    ref_text: str,
    language: str = "Auto",
    max_new_tokens: int = 512,
    batch_size: int = 32,
) -> tuple[str, list[str]]:
    model = Qwen3TTSModel.from_pretrained(
        BASELINE_MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[str] = []
    pbar = tqdm(total=len(eval_data), desc="gen-baseline")
    for i in range(0, len(eval_data), batch_size):
        sub = eval_data[i : i + batch_size]
        texts = [e["text"] for e in sub]
        langs = [language] * len(sub)
        gen_wavs_list, sr = model.generate_voice_clone(
            text=texts,
            language=langs,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            max_new_tokens=max_new_tokens,
        )
        for entry, wav in zip(sub, gen_wavs_list):
            if wav.ndim > 1:
                wav = wav.mean(axis=0)
            stem = Path(entry["audio_path"]).stem
            saved_path = str(out_dir / f"{stem}.wav")
            sf.write(saved_path, wav, sr)
            results.append(stem)
        pbar.update(len(sub))
    pbar.close()
    return str(out_dir), results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with Qwen3-TTS")
    parser.add_argument("--test-jsonl", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_all = subparsers.add_parser(
        "all", help="Generate from a fine-tuned custom-voice model"
    )
    p_all.add_argument("--model", type=str, required=True)
    p_all.add_argument("--speaker-name", type=str, required=True)

    p_base = subparsers.add_parser(
        "baseline", help="Generate via voice cloning with the base model"
    )
    p_base.add_argument("--output-path", type=Path, required=True)
    p_base.add_argument("--ref-audio", type=Path, required=True)
    p_base.add_argument("--ref-text", type=str, required=True)
    p_base.add_argument("--language", type=str, default="Auto")

    args = parser.parse_args()

    eval_data = prepare_eval_data(args.test_jsonl)
    if args.command == "all":
        saved_dir, stems = generate_all(
            args.model,
            eval_data,
            args.speaker_name,
            args.max_new_tokens,
            batch_size=args.batch_size,
        )
    elif args.command == "baseline":
        saved_dir, stems = generate_all_baseline(
            eval_data,
            args.output_path,
            args.ref_audio,
            args.ref_text,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
    else:
        parser.error(f"Unknown command: {args.command}")

    print(json.dumps({"saved_dir": saved_dir, "stems": stems}, indent=2))


if __name__ == "__main__":
    main()
