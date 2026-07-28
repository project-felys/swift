import argparse
import json
import random
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import tqdm
from transformers import AutoTokenizer

BucketConfig = tuple[tuple[float, float], ...]
BucketMap = dict[tuple[float, float], set[tuple[str, ...]]]
Stage = tuple[Sequence[float], int]
PresetResult = tuple[BucketMap, Iterator[tuple[str, ...]]]
PresetFn = Callable[[pd.DataFrame, Sequence[Sequence[int]], float], PresetResult]

SOURCE_SR = 48000
TARGET_SR = 24000


def load_language_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_language_json(path: Path):
    with open(path) as f:
        return json.load(f)


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
                entry["hash"],
                entry["name"],
                entry["text"],
                wav_duration(wav_dir / f"{entry['name']}.wav"),
                len(tokenizer.encode(entry["text"])),
            )
            for entry in tqdm.tqdm(corpus_records, desc="metadata")
        ),
        columns=["hash", "name", "text", "duration", "num_tokens"],
    )
    df = df.set_index("hash")
    df.values.flags.writeable = False
    return df


def get_bucket_range(
    duration: float, bucket_config: BucketConfig
) -> tuple[float, float] | None:
    for lo, hi in bucket_config:
        if lo <= duration < hi:
            return lo, hi
    return None


def segments_from_start(
    seq: Sequence[int],
    start: int,
    metadata: pd.DataFrame,
    gap_seconds: float,
    bucket_config: BucketConfig,
) -> dict[tuple[float, float], tuple[str, ...] | None]:
    buckets = dict.fromkeys(bucket_config, None)
    current_duration = 0.0
    current_tokens = 0

    durations = metadata["duration"]
    num_tokens = metadata["num_tokens"]
    names = metadata["name"]

    for j in range(start, len(seq)):
        h = seq[j]
        current_duration += durations[h]
        current_tokens += num_tokens[h]
        if j > start:
            current_duration += gap_seconds
            current_tokens += 1

        name = get_bucket_range(current_duration, bucket_config)
        if name is not None and buckets[name] is None and current_tokens >= 2:
            buckets[name] = tuple(names[h] for h in seq[start : j + 1])

        if name == bucket_config[-1]:
            break

    return buckets


def generate_and_bucket_segments(
    corpus: pd.DataFrame,
    sequences: Sequence[Sequence[int]],
    gap_seconds: float,
    bucket_config: BucketConfig,
) -> BucketMap:
    buckets = {key: set() for key in bucket_config}

    for seq in sequences:
        assert seq
        for start in range(len(seq)):
            result = segments_from_start(seq, start, corpus, gap_seconds, bucket_config)
            for name, segment in result.items():
                if segment is not None:
                    buckets[name].add(segment)

    return buckets


def generate_nonoverlapping_segments(
    corpus: pd.DataFrame,
    sequences: Sequence[Sequence[int]],
    gap_seconds: float,
    bucket_config: BucketConfig,
) -> BucketMap:
    buckets = {key: set() for key in bucket_config}
    durations = corpus["duration"]
    names = corpus["name"]
    upper_bound = bucket_config[-1][1]

    for seq in sequences:
        assert seq
        i = 0
        while i < len(seq):
            best_bucket: tuple[float, float] | None = None
            best_j = i
            total = 0.0
            for j in range(i, len(seq)):
                total += durations[seq[j]]
                if j > i:
                    total += gap_seconds
                b = get_bucket_range(total, bucket_config)
                if b is not None:
                    best_bucket = b
                    best_j = j
                if total >= upper_bound:
                    break
            if best_bucket is not None:
                segment = tuple(names[h] for h in seq[i : best_j + 1])
                buckets[best_bucket].add(segment)
                i = best_j + 1
            else:
                i += 1

    return buckets


def overlapping_preset(
    corpus: pd.DataFrame,
    sequences: Sequence[Sequence[int]],
    gap_seconds: float,
) -> PresetResult:
    bucket_config: BucketConfig = (
        (0.5, 5),
        (5, 12),
        (12, 25),
        (25, 45),
        (45, 90),
    )
    buckets = generate_and_bucket_segments(
        corpus, sequences, gap_seconds, bucket_config
    )
    samples = sample_stages(
        buckets,
        (
            ([0.16, 0.31, 0.30, 0.19, 0.04], 500),
            ([0.14, 0.30, 0.30, 0.20, 0.06], 500),
            ([0.13, 0.27, 0.30, 0.22, 0.08], 500),
        ),
        4,
    )
    return buckets, samples


def nonoverlapping_preset(
    corpus: pd.DataFrame,
    sequences: Sequence[Sequence[int]],
    gap_seconds: float,
) -> PresetResult:
    bucket_config: BucketConfig = (
        (0.5, 4),
        (4, 10),
        (10, 20),
        (20, 30),
    )
    buckets = generate_nonoverlapping_segments(
        corpus, sequences, gap_seconds, bucket_config
    )
    samples = sample_stages(
        buckets,
        (
            ([0.20, 0.35, 0.25, 0.20], 600),
            ([0.18, 0.32, 0.25, 0.25], 600),
        ),
        4,
    )
    return buckets, samples


PRESETS: dict[str, PresetFn] = {
    "overlapping": overlapping_preset,
    "non-overlapping": nonoverlapping_preset,
}


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


def clear_voice_resample(
    target: set[tuple[str, ...]],
    wav_dir: Path,
    output_dir: Path,
    gap_seconds: float,
) -> tuple[dict[tuple[str, ...], Path], Path]:
    from clearvoice import ClearVoice

    model = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[Path, np.ndarray] = {}

    def load(path: Path) -> np.ndarray:
        if path not in cache:
            cache[path] = load_mono(path)
        return cache[path]

    result: dict[tuple[str, ...], Path] = {}
    longest_duration = 0.0
    longest_path = None
    for segment in tqdm.tqdm(sorted(target), desc="audio"):
        out_path = output_dir / f"{compose_segment_name(segment)}.wav"
        if not out_path.exists():
            arrays = [load(wav_dir / f"{name}.wav") for name in segment]

            audio = concat_with_gap(arrays, SOURCE_SR, gap_seconds)
            with torch.no_grad():
                audio = np.asarray(model(audio[None, :]).squeeze(), dtype=np.float32)
            audio = librosa.resample(audio, orig_sr=SOURCE_SR, target_sr=TARGET_SR)

            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

        result[segment] = out_path
        duration = wav_duration(out_path)
        if duration >= longest_duration and len(segment) == 1:
            longest_duration = duration
            longest_path = out_path

    assert longest_path is not None
    return result, longest_path


def sample_stages(
    buckets: dict[tuple[float, float], set[tuple[str, ...]]],
    stages: Sequence[Stage],
    batch_size: int,
) -> Iterator[tuple[str, ...]]:
    keys = list(buckets.keys())
    pools = {k: list(v) for k, v in buckets.items()}
    for k in keys:
        random.shuffle(pools[k])
        assert batch_size <= len(pools[k])

    ptrs = {k: 0 for k in keys}

    for distribution, repeat in stages:
        assert sum(distribution) == 1
        assert len(distribution) == len(buckets)

        for _ in range(repeat):
            key = random.choices(keys, weights=distribution, k=1)[0]
            pool = pools[key]
            batch: list[tuple[str, ...]] = []
            for _ in range(batch_size):
                if ptrs[key] >= len(pool):
                    random.shuffle(pool)
                    ptrs[key] = 0
                batch.append(pool[ptrs[key]])
                ptrs[key] += 1
            yield from batch


def tokenizer_12hz_audio_codes(
    id_to_path_map: dict[tuple[str, ...], Path],
    batch_size: int,
) -> dict[tuple[str, ...], tuple[tuple[int, ...], ...]]:
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

    for path in tqdm.tqdm(id_to_path_map.values(), desc="codes"):
        buffer.append(path)
        if len(buffer) >= batch_size:
            flush()
    flush()

    return {segment: codes_by_path[path] for segment, path in id_to_path_map.items()}


def emit_for_swift_format(
    samples: Iterable[tuple[str, ...]],
    metadata: pd.DataFrame,
    audio_codes: Mapping[tuple[str, ...], tuple[tuple[int, ...], ...]],
    wav_dir: Path,
    ref_path: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    name_to_text_map = dict(zip(metadata["name"], metadata["text"]))
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in tqdm.tqdm(samples, desc="swift"):
            audio_path = wav_dir / f"{compose_segment_name(segment)}.wav"
            content = "\n".join(name_to_text_map[name] for name in segment).strip()
            record = {
                "messages": [{"role": "assistant", "content": content}],
                "audios": [str(audio_path.resolve())],
                "ref_audios": [str(ref_path.resolve())],
                "audio_codes": list(audio_codes[segment]),
            }
            print(json.dumps(record, ensure_ascii=False), file=f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bucket"),
    )
    parser.add_argument(
        "--gap-seconds",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    random.seed(args.seed)

    stem = args.dataset.name
    corpus_jsonl = args.dataset.parent / f"{stem}.jsonl"
    sequence_json = args.dataset.parent / f"{stem}.json"

    output_wav_dir = args.output_dir / "wav"
    output_ref_wav_path = args.output_dir / "ref.wav"
    output_jsonl_path = args.output_dir / f"{stem}.jsonl"

    records = load_language_jsonl(corpus_jsonl)
    sequences = load_language_json(sequence_json)
    metadata = compute_metadata(records, args.dataset)

    referenced = {h for seq in sequences for h in seq}
    sequences += [[h] for h in sorted(set(metadata.index) - referenced)]

    buckets, samples = PRESETS[args.preset](metadata, sequences, args.gap_seconds)
    all_segments = {seg for v in buckets.values() for seg in v}
    id_to_path_map, ref_path = clear_voice_resample(
        all_segments, args.dataset, output_wav_dir, args.gap_seconds
    )
    shutil.copyfile(ref_path, output_ref_wav_path)

    id_to_audio_codes = tokenizer_12hz_audio_codes(
        id_to_path_map, args.tokenizer_batch_size
    )
    emit_for_swift_format(
        samples,
        metadata,
        id_to_audio_codes,
        output_wav_dir,
        output_ref_wav_path,
        output_jsonl_path,
    )


if __name__ == "__main__":
    main()
