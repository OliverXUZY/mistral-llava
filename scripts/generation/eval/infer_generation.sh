#!/bin/bash


# Clear or create the log files
> scripts/generation/eval/infer_gene_stdout.log
> scripts/generation/eval/infer_gene_stderr.log
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


# qa_file_name="subset1K_test_generation_qa_retrie"
# as_file_name="subset1K_test_qa_retrie"

# export CUDA_VISIBLE_DEVICES=4
# model_path="liuhaotian/llava-v1.5-7b"
# num_shot=2



# export CUDA_VISIBLE_DEVICES=3
# model_path="liuhaotian/llava-v1.5-7b"
# num_shot=1


# export CUDA_VISIBLE_DEVICES=2
# model_path="SpursgoZmy/table-llava-v1.5-7b"
# num_shot=2


# export CUDA_VISIBLE_DEVICES=7
# model_path="SpursgoZmy/table-llava-v1.5-7b"
# model_path="checkpoints/table-llava-v1.5-7b-sft_rerank_02/checkpoint-4400"
# num_shot=1

######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################
######################################################################################################################################################################

export CUDA_VISIBLE_DEVICES=7
# model_path="checkpoints/llava-v1.5-7b-sft-with-table_06/checkpoint-4500"
# model_path="checkpoints/llava-v1.5-7b-sft-with-table_09/checkpoint-100"
model_path="checkpoints/llava-v1.5-7b-sft-with-table_10/checkpoint-20"
conv_template="mistral_instruct"


# model_path="SpursgoZmy/table-llava-v1.5-7b"
# conv_template="vicuna_v1"
######################################################################################################################################################################
model_filename=$(echo "$model_path" | tr '/' '_' | tr '-' '_')

# default_conversation = conv_vicuna_v1 in conversation.py so no need to specify template
######################################################################################################################################################################
######################################################################################################################################################################




##############################  single GPU, do it in a whole
python -m llava.eval.generation.infer_generation \
    --model-path "$model_path" \
    --question-file "/home/ubuntu/projects/imageTab/table_ins_ft/subset150_propor_test_generation_qa_gold.json" \
    --image-folder /home/ubuntu/projects/imageTab/ \
    --answers-file "/home/ubuntu/projects/imageTab/table_ins_ft/answers/${model_filename}/subset150_propor_test.jsonl" \
    --temperature 0 \
    --conv-mode "$conv_template" \
    >> scripts/generation/eval/infer_gene_stdout.log 2>> scripts/generation/eval/infer_gene_stderr.log

exit 0








python -m llava.eval.generation.infer_generation \
    --model-path "$model_path" \
    --question-file "/home/ubuntu/projects/imageTab/tabdata/infer_generation_testsplit/shot_${num_shot}/${qa_file_name}.json" \
    --image-folder /home/ubuntu/projects/imageTab/ \
    --answers-file "/home/ubuntu/projects/imageTab/tabdata/infer_generation_testsplit/shot_${num_shot}/answers/${model_filename}/${as_file_name}.jsonl" \
    --temperature 0 \
    --conv-mode "$conv_template"

exit 0



