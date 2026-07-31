import argparse
import json
import math
import re
from functools import cache, cached_property
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoTokenizer

TARGET_SR = 24000
N_MELS = 80
N_MFCC = 30
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_CEP = 24


class EvalNN:
    def __init__(self, ref_audio_paths: list[Path], language: str):
        self.language = language
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._stem_to_audio_path = {p.stem: p for p in ref_audio_paths}

    @cached_property
    def verification_model(self):
        from speechbrain.inference.speaker import SpeakerRecognition

        return SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )

    @cached_property
    def asr_model(self):
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B",
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="flash_attention_2",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )

    @cached_property
    def mfcc_transform(self):
        return torchaudio.transforms.MFCC(
            sample_rate=TARGET_SR,
            n_mfcc=N_MFCC,
            melkwargs={
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "n_mels": N_MELS,
            },
        )

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(
            Path(__file__).parent / "tokenizer",
            trust_remote_code=True,
            local_files_only=True,
        )

    def _load_and_normalize(
        self, audio_path: Path, target_sr: int = TARGET_SR
    ) -> torch.Tensor:
        wavs, sr = sf.read(str(audio_path), dtype="float32")
        if wavs.ndim > 1:
            wavs = wavs.mean(axis=1)
        if sr != target_sr:
            wavs = librosa.resample(wavs, orig_sr=sr, target_sr=target_sr)
        return torch.from_numpy(wavs).float()

    def _get_speaker_embedding(self, audio_path: Path) -> torch.Tensor:
        waveform = self.verification_model.load_audio(str(audio_path))
        with torch.no_grad():
            return self.verification_model.encode_batch(waveform).squeeze()

    def _extract_zcr(self, audio_path: Path) -> float:
        y = self._load_and_normalize(audio_path)
        changes = (torch.abs(torch.diff(torch.sign(y))) > 0).float()
        return float(changes.mean())

    def _text_to_tokens(self, text: str) -> list[int]:
        clean_text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text.lower())
        return self.tokenizer.encode(clean_text)

    def _transcribe(self, audio_path: Path) -> list[int]:
        results = self.asr_model.transcribe(
            audio=str(audio_path), language=self.language
        )
        return self._text_to_tokens(results[0].text)

    @cache
    def _get_embedding(self, stem: str) -> torch.Tensor:
        return self._get_speaker_embedding(self._stem_to_audio_path[stem])

    @cache
    def _get_ref_tokens(self, stem: str) -> list[int]:
        return self._transcribe(self._stem_to_audio_path[stem])

    @cache
    def _get_mfcc(self, stem: str) -> torch.Tensor:
        y_ref = self._load_and_normalize(self._stem_to_audio_path[stem])
        return self.mfcc_transform(y_ref)

    def compute_cosine_sim(self, target_path: Path) -> float:
        ref_embedding = self._get_embedding(target_path.stem)
        target_embedding = self._get_speaker_embedding(target_path)
        return F.cosine_similarity(ref_embedding, target_embedding, dim=0).item()

    def compute_zcr(self, target_path: Path) -> float:
        return self._extract_zcr(target_path)

    def compute_band_centroid(self, target_path: Path) -> dict[str, float]:
        y = self._load_and_normalize(target_path)
        S = torch.abs(torch.fft.rfft(y)) ** 2
        freqs = torch.fft.rfftfreq(len(y), 1.0 / TARGET_SR)
        total = S.sum()

        def ratio(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float((S[mask].sum() / total).item())

        centroid = float(((freqs * S).sum() / total).item())
        return {
            "band_0_300": ratio(0, 300),
            "band_300_4k": ratio(300, 4000),
            "band_4k_8k": ratio(4000, 8000),
            "band_8k_12k": ratio(8000, 12000),
            "centroid": centroid,
        }

    def compute_reference_closeness(self, target_path: Path) -> dict[str, float]:
        stem = target_path.stem
        ref_mfcc = self._get_mfcc(stem)
        y_ref = self._load_and_normalize(self._stem_to_audio_path[stem])
        y_target = self._load_and_normalize(target_path)
        target_mfcc = self.mfcc_transform(y_target)

        ref_cep = ref_mfcc[1 : N_CEP + 1].numpy()
        gen_cep = target_mfcc[1 : N_CEP + 1].numpy()

        _, path = librosa.sequence.dtw(X=ref_cep, Y=gen_cep, metric="euclidean")
        x_idx, y_idx = path[:, 0], path[:, 1]

        diffs = (ref_cep[:, x_idx].T - gen_cep[:, y_idx].T) ** 2
        mcd = float((10.0 / math.log(10.0)) * np.sqrt(2.0 * diffs.sum(axis=1)).mean())

        return {
            "mcd": mcd,
            "dur_ratio": float(len(y_target) / len(y_ref)),
        }

    def compute_wer(self, target_path: Path) -> float:
        ref_tokens = self._get_ref_tokens(target_path.stem)
        target_tokens = self._transcribe(target_path)
        return _token_edit_distance(ref_tokens, target_tokens)

    def compute_wer_batch(
        self, target_paths: list[Path], batch_size: int = 32
    ) -> dict[str, float]:
        wer_by_stem: dict[str, float] = {}
        for i in tqdm(range(0, len(target_paths), batch_size), desc="ASR batch"):
            batch = target_paths[i : i + batch_size]
            results = self.asr_model.transcribe(
                audio=[str(p) for p in batch],
                language=[self.language] * len(batch),
            )
            for path, result in zip(batch, results):
                ref_tokens = self._get_ref_tokens(path.stem)
                target_tokens = self._text_to_tokens(result.text)
                wer_by_stem[path.stem] = _token_edit_distance(ref_tokens, target_tokens)
        return wer_by_stem


def _token_edit_distance(ref: list[int], target: list[int]) -> float:
    m, n = len(ref), len(target)

    if m == 0:
        return 0.0 if n == 0 else 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / m


def evaluate_model(
    saved_dir: str,
    stems: list[str],
    evaluator: EvalNN,
) -> pd.Series:
    gen_paths = [Path(saved_dir) / f"{stem}.wav" for stem in stems]
    wer_by_stem = evaluator.compute_wer_batch(gen_paths)

    rows = []
    for gen_path, stem in tqdm(zip(gen_paths, stems), desc="per-utterance", total=len(gen_paths)):
        row = {
            "stem": stem,
            "cosine_sim": evaluator.compute_cosine_sim(gen_path),
            "wer": wer_by_stem[gen_path.stem],
            "zcr": evaluator.compute_zcr(gen_path),
            **evaluator.compute_band_centroid(gen_path),
            **evaluator.compute_reference_closeness(gen_path),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path(saved_dir).parent / "eval.csv", index=False)
    return df.mean(numeric_only=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TTS quality with neural metrics"
    )
    parser.add_argument(
        "--ground-truth-jsonl",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
    )
    args = parser.parse_args()

    with open(args.ground_truth_jsonl, encoding="utf-8") as f:
        ref_entries = [json.loads(ln) for ln in f]

    ref_audio_paths = [Path(e["audios"][0]) for e in ref_entries]
    evaluator = EvalNN(ref_audio_paths, language=args.language)

    for test_dir in args.test_dir:
        wav_paths = sorted(Path(test_dir).glob("*.wav"))
        stems = [p.stem for p in wav_paths]
        metrics = evaluate_model(test_dir, stems, evaluator)
        print(metrics)


if __name__ == "__main__":
    main()
