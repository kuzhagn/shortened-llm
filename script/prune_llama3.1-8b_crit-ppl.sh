#!/bin/bash
# All codes were adapted from https://github.com/Nota-NetsPresso/shortened-llm/tree/main
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=meta-llama/Llama-3.1-8B
export MODEL_NAME=Llama-3.1-8B
export NUM_CALIB_DATA=10
export NUM_PRUNED_BLOCKS=6
export OUTPUT_SENSITIVITY=/home/ubuntu/shortened-llm/output_block_sensitivity/$MODEL_NAME/ppl_n${NUM_CALIB_DATA}
export OUTPUT_PRUNE=/ephemeral/output_prune/$MODEL_NAME/ppl_n${NUM_CALIB_DATA}/rm_${NUM_PRUNED_BLOCKS}_blocks
export OUTPUT_TUNE=/home/ubuntu/shortened-llm//output_tune/$MODEL_NAME/ppl_n${NUM_CALIB_DATA}/rm_${NUM_PRUNED_BLOCKS}_blocks

# Analyze the PPL-based block importance with 10 calibration samples
python src/anal_block_sensitivity_ppl.py \
    --base_model $BASE_MODEL \
    --num_calib_data $NUM_CALIB_DATA \
    --output_dir $OUTPUT_SENSITIVITY

# Perform 20% block pruning by removing 6 Transformer Blocks
python src/block_prune.py \
    --base_model $BASE_MODEL \
    --num_pruned_blocks $NUM_PRUNED_BLOCKS \
    --block_order_csv $OUTPUT_SENSITIVITY/block_order.csv \
    --output_dir $OUTPUT_PRUNE

# Perform LoRA-based retraining
python src/lora_retrain.py \
    --base_model $OUTPUT_PRUNE \
    --data_path yahma/alpaca-cleaned \
    --output_dir $OUTPUT_TUNE \
    --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --save_lora_merge

# Compute Zero-shot PPL on WikiText2 and PTB 
python src/eval_ppl.py \
    --base_model ${OUTPUT_TUNE}_lora_merge_fp16 \
    --output_dir ${OUTPUT_TUNE}_score


# Serve the model (on current device)
python ray_serve_final.py 

# Estimate latency of the served model (run on other device or in the same device)
python estimate_latency.py
