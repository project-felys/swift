CUDA_VISIBLE_DEVICES=0 \
MODELSCOPE_CACHE=/root/autodl-tmp/.cache \
nohup swift sft \
    --torch_dtype 'bfloat16' \
    --model /root/autodl-tmp/pt/v9-20260611-080718/checkpoint-456 \
    --model_type 'qwen3_5' \
    --template 'qwen3_5' \
    --dataset \
        '/root/autodl-tmp/corpora/sft/cyrene' \
        '/root/autodl-tmp/corpora/sft/cyrene' \
        '/root/autodl-tmp/corpora/sft/cyrene/chs.jsonl' \
        '/root/autodl-tmp/corpora/sft/cyrene/en.jsonl' \
    --split_dataset_ratio '0.0' \
    --max_length '8192' \
    --lora_rank '16' \
    --lora_alpha '32' \
    --lora_dropout '0.01' \
    --learning_rate '1.5e-4' \
    --num_train_epochs '1.0' \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --eval_steps '50' \
    --save_steps '300' \
    --output_dir '/root/autodl-tmp/sft' \
    --attn_impl 'flash_attention_2' \
    --neftune_noise_alpha '0' \
    --truncation_strategy 'right' \
    --warmup_ratio '0.10' \
    --report_to 'tensorboard' \
    --use_liger_kernel 'True' \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --packing \
    > "$(date +%s).log" 2>&1 &
