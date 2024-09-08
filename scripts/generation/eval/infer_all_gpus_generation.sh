#!/bin/bash

# Clear or create the log files
> scripts/generation/eval/infer_gene_stdout.log
> scripts/generation/eval/infer_gene_stderr.log



# Function to run the Python script
run_python_script() {
    local device_id=$1
    local num_shot=$2
    local suffix=$3
    local model_path=$4

    export CUDA_VISIBLE_DEVICES=$device_id

    qa_file_name="subset150_propor_test_generation_qa_${suffix}"
    as_file_name="subset150_propor_test_qa_${suffix}"
    model_filename=$(echo "$model_path" | tr '/' '_' | tr '-' '_')

    python -m llava.eval.generation.infer_generation \
        --model-path "$model_path" \
        --question-file "/home/ubuntu/projects/imageTab/tabdata/infer_generation_testsplit/shot_${num_shot}/${qa_file_name}.json" \
        --image-folder /home/ubuntu/projects/imageTab/ \
        --answers-file "/home/ubuntu/projects/imageTab/tabdata/infer_generation_testsplit/shot_${num_shot}/answers/${model_filename}/${as_file_name}.jsonl" \
        --temperature 0 \
        --conv-mode "$conv_template" \
        >> scripts/generation/eval/infer_gene_stdout.log 2>> scripts/generation/eval/infer_gene_stderr.log
}

# Counter for managing GPU assignment
job_counter=0

# Array of available GPU IDs
gpu_ids=(0 1 2 3 4 5 6 7)


######################################################################################################################################################################################################
############################################################### Lists for different parameters ###############################################################
# model_path="SpursgoZmy/table-llava-v1.5-7b"
model_paths=(
    "checkpoints/llava-v1.5-7b-sft-with-table_03"
)
# model_path="checkpoints/table-llava-v1.5-7b-rerank_01"

# conv_template="vicuna_v1"
conv_template="mistral_instruct"
# List of suffixes
suffixes=("gold" "retrie" "random")
# suffixes=("rerank")

# num_shots=(1 2)
num_shots=(1)


# Iterate over parameters
for model_path in "${model_paths[@]}"; do
    echo $model_path
    echo "zhuu"
    for num_shot in "${num_shots[@]}"; do
        for suffix in "${suffixes[@]}"; do
            # Assign to GPU (assuming 8 GPUs, adjust if different)
            # device_id=$((job_counter % 8))
            # Assign to GPU (cycling through GPUs 4, 5, 6, 7)
            device_id=${gpu_ids[$((job_counter % 8))]}

            echo "Running job on GPU $device_id: num_shot=$num_shot, suffix=$suffix" >> infer_gene_stdout.log

            run_python_script $device_id $num_shot $suffix "$model_path" &

            ((job_counter++))
        done
    done
done

# Wait for all background jobs to finish
wait

echo "All jobs completed. Check stdout.log and stderr.log for results."