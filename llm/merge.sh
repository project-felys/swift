MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
swift export \
    --adapters  /root/autodl-tmp/sft/v33-20260611-125535/checkpoint-245 \
    --merge_lora true
