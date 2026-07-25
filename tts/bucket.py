import argparse
import json
import wave
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import tqdm
from transformers import AutoTokenizer

BUCKET_CONFIG: tuple[tuple[float, float], ...] = (
    (0.5, 8),
    (8, 15),
    (15, 30),
    (30, 45),
    (45, 60),
    (60, 75),
    (75, 90),
    (90, 105),
    (105, 120),
)

SOURCE_SR = 48000
TARGET_SR = 24000
CLEARVOICE_MODEL = "MossFormer2_SE_48K"


def load_language_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_language_json(path: Path):
    with open(path) as f:
        return json.load(f)


def compute_corpus_table(
    corpus_records: Sequence[Mapping[str, Any]],
    wav_dir: Path,
) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).parent / "tokenizer",
        trust_remote_code=True,
        local_files_only=True,
    )

    def wav_duration(name: str) -> float:
        with wave.open(str(wav_dir / f"{name}.wav"), "rb") as w:
            return w.getnframes() / w.getframerate()

    df = pd.DataFrame(
        data=(
            (
                entry["hash"],
                entry["name"],
                entry["text"],
                wav_duration(entry["name"]),
                len(tokenizer.encode(entry["text"])),
            )
            for entry in tqdm.tqdm(corpus_records, desc="load")
        ),
        columns=["hash", "name", "text", "duration", "num_tokens"],
    )
    df = df.set_index("hash")
    df.values.flags.writeable = False
    return df


def get_bucket_range(duration: float) -> tuple[float, float] | None:
    for lo, hi in BUCKET_CONFIG:
        if lo <= duration < hi:
            return lo, hi
    return None


def segments_from_start(
    seq: Sequence[int],
    start: int,
    corpus: pd.DataFrame,
    gap_seconds: float,
) -> dict[tuple[float, float], tuple[str, ...] | None]:
    buckets = dict.fromkeys(BUCKET_CONFIG, None)
    current_duration = 0.0
    current_tokens = 0

    durations = corpus["duration"]
    num_tokens = corpus["num_tokens"]
    names = corpus["name"]

    for j in range(start, len(seq)):
        h = seq[j]
        current_duration += durations[h]
        current_tokens += num_tokens[h]
        if j > start:
            current_duration += gap_seconds
            current_tokens += 1

        name = get_bucket_range(current_duration)
        if name is not None and buckets[name] is None and current_tokens >= 2:
            buckets[name] = tuple(names[h] for h in seq[start : j + 1])

        if name == BUCKET_CONFIG[-1]:
            break

    return buckets


def generate_and_bucket_segments(
    corpus: pd.DataFrame,
    sequences: Sequence[Sequence[int]],
    gap_seconds: float = 1.0,
) -> dict[tuple[float, float], set[tuple[str, ...]]]:
    buckets = {key: set() for key in BUCKET_CONFIG}

    for seq in sequences:
        if not seq:
            continue
        for start in range(len(seq)):
            result = segments_from_start(seq, start, corpus, gap_seconds)
            for name, segment in result.items():
                if segment is not None:
                    buckets[name].add(segment)

    return buckets


def load_mono(path: Path) -> np.ndarray:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1)


def concat_with_gap(
    arrays: Sequence[np.ndarray], sr: int, gap_seconds: float
) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    silence = np.zeros(int(sr * gap_seconds), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for i, a in enumerate(arrays):
        if i:
            chunks.append(silence)
        chunks.append(a)
    return np.concatenate(chunks)


def compose_segment_name(names: Sequence[str]) -> str:
    per_name = [n.split("_") for n in names]
    shared = 0
    for parts in zip(*per_name):
        if len({*parts}) == 1:
            shared += 1
        else:
            break
    head = per_name[0][:shared]
    tail = [p for parts in per_name for p in parts[shared:]]
    return "_".join([*head, *tail])


def enhance(audio: np.ndarray, cv) -> np.ndarray:
    if cv is None:
        return audio
    return np.asarray(cv(audio[None, :]).squeeze(), dtype=np.float32)


def clear_voice(
    target: set[tuple[str, ...]],
    wav_dir: Path,
    output_dir: Path,
    gap_seconds: float = 1.0,
) -> dict[tuple[str, ...], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from clearvoice import ClearVoice

        cv = ClearVoice(task="speech_enhancement", model_names=[CLEARVOICE_MODEL])
    except ImportError:
        cv = None

    cache: dict[Path, np.ndarray] = {}

    def load(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = load_mono(path)
        return cache[path]

    result: dict[tuple[str, ...], Path] = {}
    for segment in tqdm.tqdm(sorted(target), desc="cv"):
        out_path = output_dir / f"{compose_segment_name(segment)}.wav"
        if out_path.exists():
            result[segment] = out_path
            continue

        arrays = [load(wav_dir / f"{name}.wav") for name in segment]

        audio = concat_with_gap(arrays, SOURCE_SR, gap_seconds)
        audio = enhance(audio, cv)
        audio = librosa.resample(audio, orig_sr=SOURCE_SR, target_sr=TARGET_SR)

        sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
        result[segment] = out_path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="dataset dir holding the source wavs, e.g. corpora/tts/cyrene/Chinese(PRC)",
    )
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    args = parser.parse_args()

    stem = args.dataset.name
    corpus_jsonl = args.dataset.parent / f"{stem}.jsonl"
    sequence_json = args.dataset.parent / f"{stem}.json"

    records = load_language_jsonl(corpus_jsonl)
    sequences = load_language_json(sequence_json)
    corpus = compute_corpus_table(records, args.dataset)

    referenced = {h for seq in sequences for h in seq}
    sequences += [[h] for h in sorted(set(corpus.index) - referenced)]

    buckets = generate_and_bucket_segments(corpus, sequences, args.gap_seconds)

    for name, segs in buckets.items():
        print(f"bucket {name}: {len(segs)} segments")
    print(f"total segments: {sum(len(s) for s in buckets.values())}")


if __name__ == "__main__":
    main()
