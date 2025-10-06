#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
pred_len=96
model_name=PatchTST

root_path_name=./dataset/
entropy_model_checkpoint_dir=./entropy_model_checkpoints/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2025

# Generate all combinations and shuffle them
combinations_file="/tmp/combinations_$$"

# Generate all combinations
for heads in 2 4 8 16; do
    for layers in 1 2; do
        for max_len in 24 16 8; do
            for batch_size in 32 64 128 256; do
                for learning_rate in 0.05 0.01 0.005 0.001 0.0005 0.0001; do
                    for dim in 16 32; do
                        echo "$heads,$layers,$max_len,$batch_size,$learning_rate,$dim" >> "$combinations_file"
                    done
                done
            done
        done
    done
done

# Shuffle the combinations
shuffled_file="/tmp/shuffled_combinations_$$"
shuf "$combinations_file" > "$shuffled_file"

# Count total combinations
total_combinations=$(wc -l < "$shuffled_file")
echo "Total combinations: $total_combinations"
echo "Starting random hyperparameter search..."

# Run experiments in random order
experiment_count=0
while IFS= read -r combo; do
    experiment_count=$((experiment_count + 1))
    
    # Parse the combination
    heads=$(echo "$combo" | cut -d',' -f1)
    layers=$(echo "$combo" | cut -d',' -f2)
    max_len=$(echo "$combo" | cut -d',' -f3)
    batch_size=$(echo "$combo" | cut -d',' -f4)
    learning_rate=$(echo "$combo" | cut -d',' -f5)
    dim=$(echo "$combo" | cut -d',' -f6)
    
    echo "Experiment $experiment_count/$total_combinations"
    echo "Config: heads=$heads, layers=$layers, max_len=$max_len, batch_size=$batch_size, lr=$learning_rate, dim=$dim"
    
    # Create unique log file name with timestamp to avoid conflicts
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${experiment_count}_${timestamp}.log"
    for pred_len in 96 192 336 720
    do    
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
            --enc_in 21 \
            --vocab_size 256 \
            --quant_range 3 \
            --n_layers_local_encoder $layers \
            --n_layers_local_decoder $layers \
            --n_layers_global $layers \
            --dim_global $dim \
            --dim_local_encoder $dim \
            --dim_local_decoder $dim \
            --cross_attn_k 1 \
            --n_heads_local_encoder $heads \
            --n_heads_local_decoder $heads \
            --n_heads_global $heads \
            --cross_attn_nheads $heads \
            --cross_attn_window_encoder 96 \
            --cross_attn_window_decoder 96 \
            --local_attention_window_len 96 \
            --dropout 0.2 \
            --multiple_of 32 \
            --patch_size $max_len \
            --max_patch_length $max_len \
            --patching_threshold 0.4 \
            --patching_threshold_add 0.15 \
            --monotonicity 1 \
            --des 'Exp' \
            --train_epochs 30 \
            --patience 7 \
            --lradj 'TST' \
            --pct_start 0.3 \
            --itr 1 \
            --batch_size $batch_size \
            --patching_batch_size 896 \
            --learning_rate $learning_rate \
            >"$log_file" 2>&1
    done
    
    echo "Completed experiment $experiment_count"
    echo "Log saved to: $log_file"
    echo "----------------------------------------"
    
done < "$shuffled_file"

# Clean up temporary files
rm "$combinations_file" "$shuffled_file"

echo "All $total_combinations experiments completed in random order!"