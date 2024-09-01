#!/bin/bash

export WANDB_PROJECT=table_mistral

RUN_NUM="00"

mkdir -p "./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}"

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --version mistral_instruct \
    --data_path ./LLaVA-Finetune/enhanced_llava_sft_data_898K.json \
    --image_folder ./LLaVA-Finetune/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain-with-table_00/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "./checkpoints/llava-v1.5-7b-sft-with-table_${RUN_NUM}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5-7b-sft-with-table_${RUN_NUM}" \
    2> >(tee -a "./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}/stderr.log" >&2) | tee -a "./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}/stdout.log"

    # --report_to none \
    # -resume_from_checkpoint "checkpoints/table-llava-v1.5-7b-rerank_01/checkpoint-4700" \

    # --data_path ./data/train_rerank/train_rankRAG_qa_data.json \
    # --eval_data_path ./data/train_rerank \