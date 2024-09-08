#!/bin/bash

################################################################################################################################################
########################################################################  1.5   ################################################################################################################################################
########################################################################################################################################################################
##### vicuna 

# qa_file_name="subset1K_test_generation_qa_gold"
# as_file_name="subset1K_test_qa_gold"

# export CUDA_VISIBLE_DEVICES=1
# model_path="liuhaotian/llava-v1.5-7b"
# num_shot=2

# export CUDA_VISIBLE_DEVICES=5
# model_path="liuhaotian/llava-v1.5-7b"
# num_shot=1


# export CUDA_VISIBLE_DEVICES=6
# model_path="SpursgoZmy/table-llava-v1.5-7b"
# num_shot=1


# export CUDA_VISIBLE_DEVICES=7
# model_path="SpursgoZmy/table-llava-v1.5-7b"
# num_shot=2

######################################################################################################################################################################


qa_file_name="subset1K_test_generation_qa_retrie"
as_file_name="subset1K_test_qa_retrie"

# export CUDA_VISIBLE_DEVICES=4
# model_path="liuhaotian/llava-v1.5-7b"
# num_shot=2



# export CUDA_VISIBLE_DEVICES=3
# model_path="liuhaotian/llava-v1.5-7b"
# num_shot=1


# export CUDA_VISIBLE_DEVICES=2
# model_path="SpursgoZmy/table-llava-v1.5-7b"
# num_shot=2


export CUDA_VISIBLE_DEVICES=7
# model_path="SpursgoZmy/table-llava-v1.5-7b"
model_path="checkpoints/table-llava-v1.5-7b-sft_rerank_02/checkpoint-4400"
num_shot=1

# model_path="checkpoints/llava-v1.5-7b-rerank/checkpoint-1300"


######################################################################################################################################################################
model_filename=$(echo "$model_path" | tr '/' '_' | tr '-' '_')
conv_template="vicuna_v1"
# default_conversation = conv_vicuna_v1 in conversation.py so no need to specify template
######################################################################################################################################################################
######################################################################################################################################################################




##############################  single GPU, do it in a whole
python -m llava.eval.generation.infer_generation \
    --model-path "$model_path" \
    --question-file "/home/ubuntu/projects/imageTab/tabdata/infer_generation_testsplit/shot_${num_shot}/${qa_file_name}.json" \
    --image-folder /home/ubuntu/projects/imageTab/ \
    --answers-file "/home/ubuntu/projects/imageTab/tabdata/infer_generation_testsplit/shot_${num_shot}/answers/${model_filename}/${as_file_name}.jsonl" \
    --temperature 0 \
    --conv-mode "$conv_template"

exit 0




##############################  multiple GPUs, do it by GPU
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <device_id>"
    exit 1
fi
device_id=$1
export CUDA_VISIBLE_DEVICES=$device_id

# do it by GPU
python -m llava.eval.generation.infer_generation \
    --model-path "$model_path" \
    --question-file "/home/ubuntu/projects/imageTab/data/infer_generation_testsplit/shot_${num_shot}/${qa_file_name}.json" \
    --image-folder data/MMTab/IID_train_image \
    --answers-file "/home/ubuntu/projects/imageTab/data/infer_generation_testsplit/shot_${num_shot}/answers/${model_filename}/${as_file_name}.jsonl" \
    --temperature 0 \
    --conv-mode "$conv_template"


exit 0

