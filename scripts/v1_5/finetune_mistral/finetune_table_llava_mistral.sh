#!/bin/bash

export WANDB_PROJECT=table_mistral

RUN_NUM="10"

mkdir -p "./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}"

> ./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}/stderr.log
> ./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}/stderr.log

deepspeed llava/train/train_mem_table.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path checkpoints/llava-v1.5-7b-sft-with-table_09/checkpoint-100 \
    --version mistral_instruct \
    --data_path /home/ubuntu/projects/imageTab/table_ins_ft/subset_task_test_generation_qa_gold.json \
    --image_folder /home/ubuntu/projects/imageTab/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "./checkpoints/llava-v1.5-7b-sft-with-table_${RUN_NUM}" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --learning_rate 5e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2560 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5-7b-sft-with-table_${RUN_NUM}" \
    2> >(tee -a "./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}/stderr.log" >&2) | tee -a "./scripts/v1_5/finetune_mistral/logs/llava-v1.5-7b-sft-with-table_${RUN_NUM}/stdout.log"

    # --resume_from_checkpoint "checkpoints/llava-v1.5-7b-sft-with-table_02/checkpoint-4000" \
    # --save_total_limit 1 \

    ######################## fientune from mistral llm backbone
    # --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    # --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain-with-table_00/mm_projector.bin \
    # --data_path ./LLaVA-Finetune/enhanced_llava_sft_data_898K.json \
    # --image_folder ./LLaVA-Finetune/images \
