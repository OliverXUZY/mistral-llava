#!/bin/bash


RUN_NUM="00_debug"

mkdir -p "./scripts/v1_5/pretrain_mistral/logs/llava-v1.5-7b-pretrain_${RUN_NUM}"

> ./scripts/v1_5/pretrain_mistral/logs/pretrain/llava-v1.5-7b-pretrain_00_debug/stderr.log
> ./scripts/v1_5/pretrain_mistral/logs/pretrain/llava-v1.5-7b-pretrain_00_debug/stderr.log

deepspeed --include localhost:7 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./LLaVA-Pretrain/enhanced_llava_pretrain_data_708K.json \
    --image_folder ./LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "./checkpoints/llava-v1.5-7b-pretrain-with-table_${RUN_NUM}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2> >(tee -a "./scripts/v1_5/pretrain_mistral/logs/llava-v1.5-7b-pretrain_${RUN_NUM}/stderr.log" >&2) | tee -a "./scripts/v1_5/pretrain_mistral/logs/llava-v1.5-7b-pretrain_${RUN_NUM}/stdout.log"



