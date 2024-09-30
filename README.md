### train table-llava wirh mistral backbone
### training
see details in [google doc](https://docs.google.com/document/d/1JYuAjCbj8RsJD_S_RjH_LQAGWWHkZlOKf9Xl-hP9ams/edit?usp=sharing0).
### Reproduce with trained checkpoint: 

##### Run inference with reranking data, where data is constructed by retrieved indices.

Use train checkpoint: `"checkpoints/llava-v1.5-7b-sft-with-table_10/checkpoint-40"`

```
python scripts/generation/eval/infer_all_gpus_generation.sh
```
adjust to run full testset or subset:
```
# qa_file_name="subset100_propor_test_generation_qa_${suffix}"
# as_file_name="subset100_propor_test_qa_${suffix}"
qa_file_name="test_generation_qa_${suffix}"
as_file_name="test_qa_${suffix}"
```

get answers in `/home/ubuntu/projects/imageTab/data/infer_generation_testsplit/shot_1/stage3/answers`.

