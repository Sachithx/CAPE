#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=PatchTST

root_path_name=./dataset/
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2025

experiment_count=0

# Nested loops instead of arrays for sh compatibility
for patching_threshold in 0.25 0.3 0.35; do
    for patching_threshold_add in 0.1 0.15 0.2; do
        for learning_rate in 0.05 0.1 0.15; do
            for batch_size in 256 420 512; do
                for patching_batch_size in 2048 2940 3584; do
                    for pred_len in 96 192 336; do
                        experiment_count=$((experiment_count + 1))
                        
                        echo "Experiment $experiment_count: threshold=$patching_threshold, threshold_add=$patching_threshold_add, lr=$learning_rate, bs=$batch_size, pbs=$patching_batch_size, pred_len=$pred_len"
                        
                        timestamp=$(date +"%Y%m%d_%H%M%S")
                        log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_exp${experiment_count}_${timestamp}.log"
                    
                    python -u run_longExp.py \
                    --random_seed $random_seed \
                    --is_training 1 \
                    --root_path $root_path_name \
                    --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
                    --data_path $data_path_name \
                    --model_id ${model_id_name}_${seq_len}_${pred_len} \
                    --model_id_name $model_id_name \
                    --model $model_name \
                    --data $data_name \
                    --features M \
                    --seq_len $seq_len \
                    --pred_len $pred_len \
                    --enc_in $enc_in \
                    --vocab_size 256 \
                    --quant_range 1 \
                    --n_layers_local_encoder 2 \
                    --n_layers_local_decoder 2 \
                    --n_layers_global 2 \
                    --dim_global 32 \
                    --dim_local_encoder 32 \
                    --dim_local_decoder 32 \
                    --cross_attn_k 1 \
                    --n_heads_local_encoder 2 \
                    --n_heads_local_decoder 2 \
                    --n_heads_global 2 \
                    --cross_attn_nheads 2 \
                    --cross_attn_window_encoder 96 \
                    --cross_attn_window_decoder 96 \
                    --local_attention_window_len 96 \
                    --dropout 0.1 \
                    --multiple_of 128 \
                    --patch_size 24 \
                    --max_patch_length 24 \
                    --patching_threshold $patching_threshold \
                    --patching_threshold_add $patching_threshold_add \
                    --monotonicity 1 \
                    --des 'Exp' \
                    --train_epochs 60 \
                    --patience 25 \
                    --lradj 'TST' \
                    --pct_start 0.4 \
                    --itr 1 \
                    --batch_size $batch_size \
                    --patching_batch_size $patching_batch_size \
                    --learning_rate $learning_rate \
                    >"$log_file" 2>&1
                    
                    echo "Completed experiment $experiment_count"
                    echo "Log saved to: $log_file"
                    echo "----------------------------------------"
                done
                done
            done
        done
    done
done

echo "All experiments completed! Total: $experiment_count"