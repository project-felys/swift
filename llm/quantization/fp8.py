from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: This example requires transformers >= v5

MODEL_ID = "/root/autodl-tmp/PhiLia093-4B"

# Load model.
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_ID)


# No need to include mtp layers as they are not loaded
# through Qwen3_5ForConditionalGeneration
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_BLOCK",
        ignore=[
            "lm_head",
            "re:.*visual.*",
            "re:.*linear_attn.*",
        ],
    ),
]


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
)


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

# MTP layers are excluded from the model through Qwen3_5ForConditionalGeneration
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
