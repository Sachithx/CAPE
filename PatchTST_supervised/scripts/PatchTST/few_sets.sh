#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

seq_len=96
model_name=PatchTST
pred_len=96  # Fixed to 96

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

random_seed=2025

# Fixed parameters (do not change)
vocab_size=256
patch_size=8
max_patch_length=8
patching_threshold=0.25
patching_threshold_add=0.1
monotonicity=1

# Parameter combinations to test
n_layers_local_encoder_options="2 3 4"
n_layers_local_decoder_options="2 3 4"
n_layers_global_options="2 3 4"
dim_global_options="16 32 64"
dim_local_encoder_options="8 16 32"
dim_local_decoder_options="8 16 32"
dropout_options="0.1 0.2 0.3"
learning_rate_options="0.0001 0.0003 0.001"

# Test different parameter combinations
for n_layers_local_encoder in $n_layers_local_encoder_options
do
    for n_layers_local_decoder in $n_layers_local_decoder_options
    do
        for n_layers_global in $n_layers_global_options
        do
            for dim_global in $dim_global_options
            do
                for dim_local_encoder in $dim_local_encoder_options
                do
                    for dim_local_decoder in $dim_local_decoder_options
                    do
                        for dropout in $dropout_options
                        do
                            for learning_rate in $learning_rate_options
                            do
                                # Create unique model identifier for this parameter combination
                                param_id="le${n_layers_local_encoder}_ld${n_layers_local_decoder}_lg${n_layers_global}_dg${dim_global}_dle${dim_local_encoder}_dld${dim_local_decoder}_dr${dropout}_lr${learning_rate}"
                                
                                echo "Running experiment with parameters: ${param_id}"
                                
                                python -u run_longExp.py \
                                  --random_seed $random_seed \
                                  --is_training 1 \
                                  --root_path $root_path_name \
                                  --data_path $data_path_name \
                                  --model_id ${model_id_name}_${seq_len}_${pred_len}_${param_id} \
                                  --model $model_name \
                                  --data $data_name \
                                  --features M \
                                  --seq_len $seq_len \
                                  --pred_len $pred_len \
                                  --enc_in 7 \
                                  --vocab_size $vocab_size \
                                  --quant_range 6 \
                                  --n_layers_local_encoder $n_layers_local_encoder \
                                  --n_layers_local_decoder $n_layers_local_decoder \
                                  --n_layers_global $n_layers_global \
                                  --dim_global $dim_global \
                                  --dim_local_encoder $dim_local_encoder \
                                  --dim_local_decoder $dim_local_decoder \
                                  --cross_attn_k 1 \
                                  --n_heads_local_encoder 2 \
                                  --n_heads_local_decoder 2 \
                                  --n_heads_global 4 \
                                  --cross_attn_nheads 2 \
                                  --cross_attn_window_encoder 96 \
                                  --cross_attn_window_decoder 96 \
                                  --local_attention_window_len 96 \
                                  --dropout $dropout \
                                  --patch_size $patch_size \
                                  --max_patch_length $max_patch_length \
                                  --patching_threshold $patching_threshold \
                                  --patching_threshold_add $patching_threshold_add \
                                  --monotonicity $monotonicity \
                                  --des 'Exp' \
                                  --train_epochs 180 \
                                  --patience 100 \
                                  --lradj 'TST' \
                                  --pct_start 0.4 \
                                  --itr 1 \
                                  --batch_size 128 \
                                  --patching_batch_size 128 \
                                  --learning_rate $learning_rate \
                                  >logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_${param_id}.log 
                            done
                        done
                    done
                done
            done
        done
    done
done