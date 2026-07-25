import argparse

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS custom voice inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF repo id of the fine-tuned checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--text", type=str, default="你是银河猫猫侠深爱的昔涟，正陪着她聊天。"
    )
    parser.add_argument("--speaker", type=str, default="cyrene")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max codec frames to generate (~audio_seconds * 12). Lower it if generation hangs.",
    )
    args = parser.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        max_new_tokens=args.max_new_tokens,
    )

    sf.write(args.output, wavs[0], sr)


if __name__ == "__main__":
    main()
