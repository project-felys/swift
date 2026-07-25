import argparse
import json
from functools import cache, cached_property
from pathlib import Path
import re

import librosa
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoTokenizer

from tqdm import tqdm

TARGET_SR = 24000
N_MELS = 80
N_MFCC = 30
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024


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
    def utmos_model(self):
        import utmosv2

        return utmosv2.create_model(pretrained=True).to(self.device)

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
    def mel_transform(self):
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            power=2.0,
            normalized=True,
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

    def _load_and_normalize(self, audio_path: Path) -> torch.Tensor:
        wavs, sr = sf.read(str(audio_path), dtype="float32")
        if wavs.ndim > 1:
            wavs = wavs.mean(axis=1)
        if sr != TARGET_SR:
            wavs = librosa.resample(wavs, orig_sr=sr, target_sr=TARGET_SR)
        return torch.from_numpy(wavs).float()

    def _get_speaker_embedding(self, audio_path: Path) -> torch.Tensor:
        waveform = self.verification_model.load_audio(str(audio_path))
        with torch.no_grad():
            return self.verification_model.encode_batch(waveform).squeeze()

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
    def _get_mel_spec(self, stem: str) -> torch.Tensor:
        y_ref = self._load_and_normalize(self._stem_to_audio_path[stem])
        return self.mel_transform(y_ref)

    @cache
    def _get_mfcc(self, stem: str) -> torch.Tensor:
        y_ref = self._load_and_normalize(self._stem_to_audio_path[stem])
        return self.mfcc_transform(y_ref)

    @cache
    def _get_utmos(self, stem: str) -> float:
        return self.utmos_model.predict(input_path=str(self._stem_to_audio_path[stem]))

    def compute_cosine_sim(self, target_path: Path) -> float:
        ref_embedding = self._get_embedding(target_path.stem)
        target_embedding = self._get_speaker_embedding(target_path)
        return F.cosine_similarity(ref_embedding, target_embedding, dim=0).item()

    def compute_ref_utmos_by_stem(self, stem) -> float:
        return self._get_utmos(stem)

    def compute_utmos(self, audio_path: Path) -> float:
        return self.utmos_model.predict(input_path=str(audio_path))

    def compute_utmos_batch(self, audio_dir: Path) -> dict[str, float]:
        results = self.utmos_model.predict(input_dir=str(audio_dir))
        return {Path(r["file_path"]).stem: r["predicted_mos"] for r in results}

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

    def compute_mel_cosine(self, target_path: Path) -> float:
        ref_mel = self._get_mel_spec(target_path.stem)
        y_target = self._load_and_normalize(target_path)
        target_mel = self.mel_transform(y_target)
        return _feature_cosine_sim(ref_mel, target_mel)

    def compute_mfcc_cosine(self, target_path: Path) -> float:
        ref_mfcc = self._get_mfcc(target_path.stem)
        y_target = self._load_and_normalize(target_path)
        target_mfcc = self.mfcc_transform(y_target)
        return _feature_cosine_sim(ref_mfcc, target_mfcc)


def _feature_cosine_sim(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    vec_a = feat_a.mean(dim=-1)
    vec_b = feat_b.mean(dim=-1)
    return F.cosine_similarity(vec_a, vec_b, dim=0).item()


def _token_edit_distance(ref: list[int], target: list[int]) -> float:
    m, n = len(ref), len(target)

    if m == 0:
        return float(n)

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
    utmos_by_stem = evaluator.compute_utmos_batch(Path(saved_dir))
    wer_by_stem = evaluator.compute_wer_batch(gen_paths)

    rows = []
    for gen_path, stem in zip(gen_paths, stems):
        utmos = utmos_by_stem[gen_path.stem]
        ref_utmos = evaluator.compute_ref_utmos_by_stem(gen_path.stem)
        row = {
            "stem": stem,
            "cosine_sim": evaluator.compute_cosine_sim(gen_path),
            "utmos": utmos,
            "ref_utmos": ref_utmos,
            "wer": wer_by_stem[gen_path.stem],
            "mel_cosine": evaluator.compute_mel_cosine(gen_path),
            "mfcc_cosine": evaluator.compute_mfcc_cosine(gen_path),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path(saved_dir).parent / "eval.csv", index=False)
    df = df.assign(utmos_mse=(df["utmos"] - df["ref_utmos"]) ** 2)
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

    ref_entries = [
        json.loads(ln) for ln in open(args.ground_truth_jsonl, encoding="utf-8")
    ]

    ref_audio_paths = [Path(e["audios"][0]) for e in ref_entries]
    evaluator = EvalNN(ref_audio_paths, language=args.language)

    for test_dir in args.test_dir:
        wav_paths = sorted(Path(test_dir).glob("*.wav"))
        stems = [p.stem for p in wav_paths]
        metrics = evaluate_model(test_dir, stems, evaluator)
        print(metrics)


if __name__ == "__main__":
    main()
