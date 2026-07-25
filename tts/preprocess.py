from __future__ import annotations

import argparse
import json
import random
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
)

import librosa
import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from transformers import AutoTokenizer

SOURCE_SR = 48000
TARGET_SR = 24000
CLEARVOICE_MODEL = "MossFormer2_SE_48K"
TOKENIZER_BATCH = 32
DEFAULT_WINDOWS = (1, 2, 4, 8, 16, 32)
DEFAULT_SEED = 42
DEFAULT_MIN_SECONDS = 0.5
DEFAULT_MIN_TOKENS = 2
DEFAULT_TEST_RATIO = 0.10
DEFAULT_N_REPEATS = 3
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_GAP_SECONDS = 1.0
DEFAULT_MAX_PER_AUDIO = 5


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_jsonl(), ensure_ascii=False) + "\n")
            n += 1
    return n


def _read_clip_mono(path: Path, sr: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(audio, dtype=np.float32)


def enhance_audio(audio: np.ndarray, cv) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    assert x.ndim == 1
    with torch.no_grad():
        out = cv(x[None, :]).squeeze()
    return np.asarray(out, dtype=np.float32)


def make_token_counter() -> Callable[[str], int]:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).parent / "tokenizer",
        trust_remote_code=True,
        local_files_only=True,
    )
    return lambda text: len(tokenizer.encode(text))


def compose_clip_name(stems: Sequence[str]) -> str:
    if len(stems) == 1:
        return stems[0]
    token_lists = [s.split("_") for s in stems]
    prefix_len = 0
    for tokens in zip(*token_lists):
        if len(set(tokens)) == 1:
            prefix_len += 1
        else:
            break
    prefix = token_lists[0][:prefix_len]
    suffixes = [tok for tl in token_lists for tok in tl[prefix_len:]]
    return "_".join(prefix + suffixes)


class SourceClip(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    hash: int
    text: str

    @classmethod
    def from_jsonl(cls, data: Mapping[str, Any]) -> "SourceClip":
        return cls(name=data["name"], hash=int(data["hash"]), text=data["text"])


class EnhancedClip(BaseModel):
    model_config = ConfigDict(frozen=True)

    audio: str
    text: str
    sources: Tuple[int, ...]

    def to_jsonl(self) -> Mapping[str, Any]:
        return {"audio": self.audio, "text": self.text, "sources": list(self.sources)}


class SwiftRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    audios: Tuple[str, ...]
    ref_audios: Tuple[str, ...]
    sources: Tuple[int, ...]

    def to_jsonl(self) -> Mapping[str, Any]:
        return {
            "messages": [{"role": "assistant", "content": self.content}],
            "audios": list(self.audios),
            "ref_audios": list(self.ref_audios),
            "sources": list(self.sources),
        }


class CodedRecord(SwiftRecord):
    audio_codes: Tuple[Tuple[int, ...], ...]

    def to_jsonl(self) -> Mapping[str, Any]:
        return {**super().to_jsonl(), "audio_codes": list(self.audio_codes)}

    def stripped_line(self) -> str:
        return json.dumps(
            {
                "messages": [{"role": "assistant", "content": self.content}],
                "audios": list(self.audios),
                "ref_audios": list(self.ref_audios),
                "audio_codes": list(self.audio_codes),
            },
            ensure_ascii=False,
        )


class CorpusLoader:
    def __init__(
        self,
        corpus_jsonl: Path,
        sequence_json: Path,
        window_sizes: Sequence[int] = DEFAULT_WINDOWS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._corpus_jsonl = corpus_jsonl
        self._sequence_json = sequence_json
        self._window_sizes = tuple(window_sizes)
        self._seed = seed

    @cached_property
    def source_clips(self) -> Mapping[int, SourceClip]:
        return {
            clip.hash: clip
            for clip in (
                SourceClip.from_jsonl(d) for d in _read_jsonl(self._corpus_jsonl)
            )
        }

    @cached_property
    def _sequences(self) -> list[list[int]]:
        known = self.source_clips
        raw = json.loads(self._sequence_json.read_text(encoding="utf-8"))
        assert isinstance(raw, list), "sequence json must be a list[list[int]]"
        sequences: list[list[int]] = []
        missing = 0
        seen: set[int] = set()
        for seq in raw:
            assert isinstance(seq, list), "sequence json must be list[list[int]]"
            kept = [int(h) for h in seq if int(h) in known]
            missing += len(seq) - len(kept)
            seen.update(kept)
            if kept:
                sequences.append(kept)
        orphans = sorted(known.keys() - seen)
        for h in orphans:
            sequences.append([h])
        print(
            f"[sequence] {len(sequences)} sequences from {len(known)} source clips "
            f"({missing} missing hashes dropped, {len(orphans)} orphan singletons added)"
        )
        return sequences

    @cached_property
    def batches(self) -> list[tuple[int, ...]]:
        rng = random.Random(self._seed)
        all_batches: list[tuple[int, ...]] = []
        for window in self._window_sizes:
            assert window > 0
            window_batches: list[tuple[int, ...]] = []
            for seq in self._sequences:
                for i in range(0, len(seq), window):
                    piece = seq[i : i + window]
                    if piece:
                        window_batches.append(tuple(piece))
            rng.shuffle(window_batches)
            all_batches.extend(window_batches)
        print(f"[batch] {len(all_batches)} raw batches")
        return all_batches


class EnhanceStage:
    def __init__(
        self,
        source_wav_dir: Path,
        out_dir: Path,
        stem: str,
        min_seconds: float = DEFAULT_MIN_SECONDS,
        min_tokens: int = DEFAULT_MIN_TOKENS,
        force: bool = False,
        max_per_audio: int = DEFAULT_MAX_PER_AUDIO,
        seed: int = DEFAULT_SEED,
    ) -> None:
        from clearvoice import ClearVoice

        self._source_wav_dir = source_wav_dir
        self._out_dir = out_dir
        self._stem = stem
        self._cv = ClearVoice(task="speech_enhancement", model_names=[CLEARVOICE_MODEL])
        self._token_counter = make_token_counter()
        self._min_seconds = min_seconds
        self._min_tokens = min_tokens
        self._force = force
        self._max_per_audio = max_per_audio
        self._seed = seed

    @property
    def enhanced_dir(self) -> Path:
        return self._out_dir / self._stem

    @property
    def manifest_path(self) -> Path:
        return self._out_dir / f"{self._stem}.jsonl"

    def _concat_with_gap(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        if len(arrays) == 1:
            return arrays[0]
        gap = np.zeros(int(SOURCE_SR * DEFAULT_GAP_SECONDS), dtype=np.float32)
        parts: list[np.ndarray] = []
        for i, a in enumerate(arrays):
            if i > 0:
                parts.append(gap)
            parts.append(a)
        return np.concatenate(parts)

    def _downsample(
        self, records: list[EnhancedClip], max_per_audio: int
    ) -> list[EnhancedClip]:
        if max_per_audio <= 0:
            return records
        name_to_indices: dict[str, set[int]] = {}
        for i, rec in enumerate(records):
            name_to_indices.setdefault(rec.audio, set()).add(i)
        to_remove: set[int] = set()
        rng = random.Random(self._seed)
        for indices in name_to_indices.values():
            if len(indices) > max_per_audio:
                excess = len(indices) - max_per_audio
                candidates = sorted(indices)
                rng.shuffle(candidates)
                to_remove.update(candidates[:excess])
        kept = [r for i, r in enumerate(records) if i not in to_remove]
        dropped = len(records) - len(kept)
        if dropped:
            print(
                f"[downsample] dropped {dropped} records ({max_per_audio=}) -> {len(kept)}"
            )
        return kept

    def run(
        self,
        batches: Sequence[tuple[int, ...]],
        source_clips: Mapping[int, SourceClip],
    ) -> list[EnhancedClip]:
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)

        clip_cache: dict[Path, np.ndarray] = {}
        records: list[EnhancedClip] = []
        skipped_dur = skipped_tok = 0

        def load_clip(path: Path) -> np.ndarray:
            if path not in clip_cache:
                clip_cache[path] = _read_clip_mono(path, SOURCE_SR)
            return clip_cache[path]

        # Pass 1: filter + gather records (no CV)
        for batch in tqdm(batches, desc="filter", unit="batch"):
            clips = [source_clips[int(h)] for h in batch]
            arrays = [load_clip(self._source_wav_dir / f"{c.name}.wav") for c in clips]
            if sum(len(a) for a in arrays) / SOURCE_SR < self._min_seconds:
                skipped_dur += 1
                continue

            text = "\n".join(c.text for c in clips)
            if self._token_counter(text) < self._min_tokens:
                skipped_tok += 1
                continue

            audio_name = f"{compose_clip_name([c.name for c in clips])}.wav"
            records.append(EnhancedClip(audio=audio_name, text=text, sources=batch))

        # Downsample before CV (save compute on dropped records)
        records = self._downsample(records, self._max_per_audio)

        # Pass 2: CV + write wavs (pooled by audio_name — file already exists -> skip)
        for rec in tqdm(records, desc="cv", unit="wav"):
            out_path = self.enhanced_dir / rec.audio
            if self._force or not out_path.exists():
                clips = [source_clips[int(h)] for h in rec.sources]
                arrays = [
                    clip_cache[self._source_wav_dir / f"{c.name}.wav"] for c in clips
                ]
                audio = self._concat_with_gap(arrays)
                sf.write(str(out_path), enhance_audio(audio, self._cv), SOURCE_SR)

        n = _write_jsonl(self.manifest_path, records)
        print(
            f"[enhance] wrote {n} clips -> {self.manifest_path} | "
            f"skipped dur<{self._min_seconds}s={skipped_dur}, "
            f"tok<{self._min_tokens}={skipped_tok}"
        )
        return records


class SwiftStage:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def _pick_ref(
        self, enhanced: Sequence[EnhancedClip], enhanced_wav_dir: Path
    ) -> tuple[str, float]:
        pool = [r for r in enhanced if len(r.sources) == 1]
        assert pool, "no single-clip (original) enhanced record to use as ref"
        best, best_dur = pool[0], -1.0
        for r in pool:
            info = sf.info(str(enhanced_wav_dir / r.audio))
            dur = info.frames / info.samplerate
            if dur > best_dur:
                best, best_dur = r, dur
        return best.audio, best_dur

    def run(
        self,
        enhanced: Sequence[EnhancedClip],
        enhanced_wav_dir: Path,
        stem: str,
    ) -> list[SwiftRecord]:
        wav_dir = self._out_dir / stem
        out_jsonl = self._out_dir / f"{stem}.jsonl"
        ref_path = self._out_dir / "ref.wav"
        wav_dir.mkdir(parents=True, exist_ok=True)

        # Resample each unique enhanced wav once (records may share an audio).
        unique_audios = sorted({e.audio for e in enhanced})
        for audio in tqdm(unique_audios, desc="swift", unit="wav"):
            y, _ = librosa.load(str(enhanced_wav_dir / audio), sr=TARGET_SR, mono=True)
            sf.write(str(wav_dir / audio), y, TARGET_SR, subtype="PCM_16")

        # Reference: reuse an already-enhanced wav, same enhance->resample path.
        ref_name, ref_dur = self._pick_ref(enhanced, enhanced_wav_dir)
        ref_y, _ = librosa.load(
            str(enhanced_wav_dir / ref_name), sr=TARGET_SR, mono=True
        )
        sf.write(str(ref_path), ref_y, TARGET_SR, subtype="PCM_16")

        records: list[SwiftRecord] = []
        for clip in enhanced:
            dst = wav_dir / clip.audio
            records.append(
                SwiftRecord(
                    content=clip.text.strip(),
                    audios=(str(dst.resolve()),),
                    ref_audios=(str(ref_path.resolve()),),
                    sources=clip.sources,
                )
            )

        n = _write_jsonl(out_jsonl, records)
        print(
            f"[swift] wrote {n} -> {out_jsonl} | ref -> {ref_path} "
            f"({ref_name}, {ref_dur:.2f}s)"
        )
        return records


class CodesStage:
    def __init__(self, out_dir: Path, device: str, model_path: str) -> None:
        self._out_dir = out_dir
        self._device = device
        self._model_path = model_path

    def run(
        self,
        records: Sequence[SwiftRecord],
        name: str,
    ) -> list[CodedRecord]:
        from qwen_tts import Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            self._model_path, device_map=self._device
        )
        out_path = self._out_dir / name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Group records by unique audio path (pooling: same wav -> same codes)
        path_to_records: dict[str, list[SwiftRecord]] = {}
        for rec in records:
            path_to_records.setdefault(rec.audios[0], []).append(rec)
        unique_paths = sorted(path_to_records)

        # Encode each unique audio once
        path_to_codes: dict[str, tuple[tuple[int, ...], ...]] = {}
        buf_paths: list[str] = []

        def flush() -> None:
            if not buf_paths:
                return
            res = tokenizer.encode(buf_paths)
            for p, code in zip(buf_paths, res.audio_codes):
                path_to_codes[p] = tuple(
                    tuple(int(x) for x in row) for row in code.cpu().tolist()
                )
            buf_paths.clear()

        for p in tqdm(unique_paths, desc="codes", unit="wav"):
            buf_paths.append(p)
            if len(buf_paths) >= TOKENIZER_BATCH:
                flush()
        flush()

        # Build coded records (reuse cached codes)
        coded: list[CodedRecord] = []
        for rec in records:
            coded.append(
                CodedRecord(
                    content=rec.content,
                    audios=rec.audios,
                    ref_audios=rec.ref_audios,
                    sources=rec.sources,
                    audio_codes=path_to_codes[rec.audios[0]],
                )
            )

        n = _write_jsonl(out_path, coded)
        print(
            f"[codes] wrote {n} -> {out_path} | unique audios encoded: {len(path_to_codes)}"
        )
        return coded


class SplitStage:
    def __init__(
        self,
        test_ratio: float = DEFAULT_TEST_RATIO,
        n_repeats: int = DEFAULT_N_REPEATS,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self._test_ratio = test_ratio
        self._n_repeats = n_repeats
        self._seed = seed

    def run(self, records: Sequence[CodedRecord], out_dir: Path, name: str) -> None:
        n = len(records)
        target_test = max(1, round(n * self._test_ratio)) if records else 0

        clip_to_records: dict[int, set[int]] = {}
        for i, rec in enumerate(records):
            for clip in rec.sources:
                clip_to_records.setdefault(clip, set()).add(i)
        all_clips = list(clip_to_records.keys())

        rng = random.Random(self._seed)
        for shard in range(self._n_repeats):
            rng.shuffle(all_clips)
            test_idx: set[int] = set()
            for clip in all_clips:
                if len(test_idx) >= target_test:
                    break
                test_idx |= clip_to_records[clip]
            train_idx = [i for i in range(n) if i not in test_idx]

            train_out = out_dir / "train" / str(shard) / name
            test_out = out_dir / "test" / str(shard) / name
            train_out.parent.mkdir(parents=True, exist_ok=True)
            test_out.parent.mkdir(parents=True, exist_ok=True)

            train_out.write_text(
                "".join(records[i].stripped_line() + "\n" for i in train_idx),
                encoding="utf-8",
            )
            test_out.write_text(
                "".join(records[i].stripped_line() + "\n" for i in sorted(test_idx)),
                encoding="utf-8",
            )

            test_clips = sum(1 for c in all_clips if clip_to_records[c] <= test_idx)
            print(
                f"[split] shard {shard}: train {len(train_idx)} | "
                f"test {len(test_idx)} (target {target_test}, "
                f"~{test_clips} test clips of {len(all_clips)})"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="dataset dir, e.g. corpora/tts/cyrene/Chinese(PRC)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--window-sizes",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WINDOWS),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument(
        "--max-per-audio",
        type=int,
        default=DEFAULT_MAX_PER_AUDIO,
        help="max records per unique audio name (downsample)",
    )
    parser.add_argument("--force", action="store_true", help="re-run ClearVoice")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--tokenizer-model-path", type=str, default=DEFAULT_TOKENIZER_MODEL
    )
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS)
    args = parser.parse_args(argv)

    stem = args.dataset.name
    args.stem = stem
    args.wav_dir = args.dataset
    args.corpus_jsonl = args.dataset.parent / f"{stem}.jsonl"
    args.sequence_json = args.dataset.parent / f"{stem}.json"
    args.window_sizes = tuple(int(w) for w in args.window_sizes.split(",") if w.strip())
    return args


def main() -> None:
    args = parse_args()

    assert args.corpus_jsonl.is_file(), f"corpus jsonl not found: {args.corpus_jsonl}"
    assert args.sequence_json.is_file(), (
        f"sequence json not found: {args.sequence_json}"
    )
    assert args.wav_dir.is_dir(), f"wav dir not found: {args.wav_dir}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = CorpusLoader(
        corpus_jsonl=args.corpus_jsonl,
        sequence_json=args.sequence_json,
        window_sizes=args.window_sizes,
        seed=args.seed,
    )

    enhance = EnhanceStage(
        source_wav_dir=args.wav_dir,
        out_dir=args.output_dir / "enhanced",
        stem=args.stem,
        min_seconds=args.min_seconds,
        min_tokens=args.min_tokens,
        force=args.force,
        max_per_audio=args.max_per_audio,
        seed=args.seed,
    )
    enhanced = enhance.run(loader.batches, loader.source_clips)

    swift = SwiftStage(out_dir=args.output_dir / "raw")
    swift_records = swift.run(
        enhanced, enhanced_wav_dir=enhance.enhanced_dir, stem=args.stem
    )

    codes = CodesStage(
        out_dir=args.output_dir,
        device=args.device,
        model_path=args.tokenizer_model_path,
    )
    coded_name = f"{args.stem}.jsonl"
    coded = codes.run(swift_records, name=coded_name)

    SplitStage(
        test_ratio=args.test_ratio,
        n_repeats=args.n_repeats,
        seed=args.seed,
    ).run(coded, out_dir=args.output_dir, name=coded_name)


if __name__ == "__main__":
    main()
