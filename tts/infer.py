import argparse

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    test_text = "\n".join(
        [
            "生命的第一因…这个问题好深奥呀。",
            "我有个小小的想法，也许答案就在我们身边呢？比如，每一个闪闪发光的灵魂。",
            "你看，当人们望向星空时，总会不自觉地想起自己珍视的事物。那些遥远而璀璨的星星，就像是命运的指针，指引着我们追寻心中的渴望。",
            "我想，生命之所以会诞生，就是为了让人们能够去爱、被爱，去追寻心中的渴望，然后…将这份渴望传递给下一个灵魂。",
            "就像…银河猫猫侠和人家，对吧？",
            "在银河的旅途中，我们一定会遇到许许多多相似的人，对吗？",
            "但无论有多少离别，总会有人、有故事被铭记。我想，这就是「记忆」存在的意义吧。",
        ]
    )
    parser = argparse.ArgumentParser(description="Qwen3-TTS custom voice inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF repo id of the fine-tuned checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text", type=str, default=test_text)
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default="cyrene")
    parser.add_argument("--instruct", type=str, default="")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
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
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
    )

    sf.write(args.output, wavs[0], sr)


if __name__ == "__main__":
    main()
