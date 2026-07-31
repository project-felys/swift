import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
from typing import Any

import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
from transformers import AutoTokenizer

SOURCE_SR = 48000
TARGET_SR = 24000


def load_language_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def compute_metadata(
    corpus_records: Sequence[Mapping[str, Any]],
    wav_dir: Path,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).parent / "tokenizer",
        trust_remote_code=True,
        local_files_only=True,
    )

    df = pd.DataFrame(
        data=(
            (
                entry["name"],
                entry["text"],
                wav_duration(wav_dir / f"{entry['name']}.wav"),
                len(tokenizer.encode(entry["text"])),
            )
            for entry in tqdm.tqdm(corpus_records, desc="metadata")
        ),
        columns=["name", "text", "duration", "num_tokens"],
    )
    df.values.flags.writeable = False
    return df


def load_mono(path: Path) -> np.ndarray:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1)


def clear_voice_resample(
    metadata: pd.DataFrame,
    wav_dir: Path,
    output_dir: Path,
    min_tokens: int,
    min_seconds: float,
) -> tuple[dict[str, Path], Path]:
    from clearvoice import ClearVoice

    model = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    output_dir.mkdir(parents=True, exist_ok=True)

    name_to_path: dict[str, Path] = {}
    longest_duration = 0.0
    longest_path = None

    for row in tqdm.tqdm(
        metadata.itertuples(index=False), desc="audio", total=len(metadata)
    ):
        if row.num_tokens < min_tokens:
            continue
        if row.duration < min_seconds:
            continue

        out_path = output_dir / f"{row.name}.wav"
        if not out_path.exists():
            audio = load_mono(wav_dir / f"{row.name}.wav")
            with torch.no_grad():
                audio = np.asarray(model(audio[None, :]).squeeze(), dtype=np.float32)
            audio = librosa.resample(audio, orig_sr=SOURCE_SR, target_sr=TARGET_SR)
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

        duration = wav_duration(out_path)
        if duration >= longest_duration:
            longest_duration = duration
            longest_path = out_path

        name_to_path[row.name] = out_path

    assert longest_path is not None
    return name_to_path, longest_path


def tokenizer_12hz_audio_codes(
    name_to_path: dict[str, Path],
    batch_size: int,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    from qwen_tts import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0"
    )

    codes_by_path: dict[Path, tuple[tuple[int, ...], ...]] = {}
    buffer: list[Path] = []

    def flush() -> None:
        if not buffer:
            return
        res = tokenizer.encode([str(p) for p in buffer])
        for path, code in zip(buffer, res.audio_codes):
            codes_by_path[path] = tuple(
                tuple(int(x) for x in row) for row in code.cpu().tolist()
            )
        buffer.clear()

    for path in tqdm.tqdm(name_to_path.values(), desc="codes"):
        buffer.append(path)
        if len(buffer) >= batch_size:
            flush()
    flush()

    return {name: codes_by_path[path] for name, path in name_to_path.items()}


def build_ordered_names(
    names: Sequence[str],
    num_epochs: int,
    seed: int,
) -> list[str]:
    ordered: list[str] = []
    for epoch in range(num_epochs):
        epoch_rng = random.Random(seed + epoch)
        epoch_names = list(names)
        epoch_rng.shuffle(epoch_names)
        ordered.extend(epoch_names)
    return ordered


def emit_for_swift_format(
    names: Sequence[str],
    metadata: pd.DataFrame,
    audio_codes: Mapping[str, tuple[tuple[int, ...], ...]],
    wav_dir: Path,
    ref_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    name_to_text = dict(zip(metadata["name"], metadata["text"]))
    with open(output_path, "w", encoding="utf-8") as f:
        for name in tqdm.tqdm(names, desc="swift"):
            audio_path = wav_dir / f"{name}.wav"
            content = name_to_text[name]
            record = {
                "messages": [{"role": "assistant", "content": content}],
                "audios": [str(audio_path.resolve())],
                "ref_audios": [str(ref_path.resolve())],
                "audio_codes": list(audio_codes[name]),
            }
            print(json.dumps(record, ensure_ascii=False), file=f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audio"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-batch-size", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--min-seconds", type=float, default=2)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--num-splits", type=int, default=3)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    stem = args.dataset.name
    corpus_jsonl = args.dataset.parent / f"{stem}.jsonl"

    output_wav_dir = args.output_dir / "wav"
    output_ref_wav_path = args.output_dir / "ref.wav"
    output_jsonl_path = args.output_dir / f"{stem}.jsonl"

    records = load_language_jsonl(corpus_jsonl)
    metadata = compute_metadata(records, args.dataset)

    name_to_path, ref_path = clear_voice_resample(
        metadata, args.dataset, output_wav_dir, args.min_tokens, args.min_seconds
    )
    shutil.copyfile(ref_path, output_ref_wav_path)

    name_to_codes = tokenizer_12hz_audio_codes(name_to_path, args.tokenizer_batch_size)

    names = list(name_to_path.keys())
    ordered_names = build_ordered_names(names, args.num_epochs, args.seed)

    emit_for_swift_format(
        ordered_names,
        metadata,
        name_to_codes,
        output_wav_dir,
        output_ref_wav_path,
        output_jsonl_path,
    )

    train_dir = args.output_dir / "train"
    test_dir = args.output_dir / "test"
    num_test = max(1, int(len(names) * args.test_ratio))

    for split in range(args.num_splits):
        split_rng = random.Random(args.seed + split)
        split_names = list(names)
        split_rng.shuffle(split_names)
        test_names = split_names[:num_test]
        train_names = split_names[num_test:]

        train_ordered = build_ordered_names(
            train_names, args.num_epochs, args.seed
        )
        emit_for_swift_format(
            train_ordered,
            metadata,
            name_to_codes,
            output_wav_dir,
            output_ref_wav_path,
            train_dir / f"{split}.jsonl",
        )
        emit_for_swift_format(
            test_names,
            metadata,
            name_to_codes,
            output_wav_dir,
            output_ref_wav_path,
            test_dir / f"{split}.jsonl",
        )


if __name__ == "__main__":
    main()
