
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /path/to/your/base_model \
    --train_type lora \
    --dataset ./data/sft.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /path/to/save/results \
    --warmup_ratio 0 \
    --dataloader_num_workers 0 \


# v1 使用相同label dpo