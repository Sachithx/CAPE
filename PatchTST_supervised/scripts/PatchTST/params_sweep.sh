#!/bin/bash

# Create log directories
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/HyperparamSearch" ]; then
    mkdir ./logs/HyperparamSearch
fi

# Fixed parameters for 96->96 prediction
seq_len=96
pred_len=96
model_name=PatchTST
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
random_seed=2025

echo "=== PatchTST Hyperparameter Exploration for ETTm1 (96->96) ==="
echo "Starting hyperparameter search..."

# Experiment 1: Different Patch Sizes (Critical for PatchTST)
echo "Experiment 1: Testing different patch sizes..."
for patch_size in 4 8 12 16 24
do
    echo "Running with patch_size=$patch_size"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_patch${patch_size}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --patch_size $patch_size \
      --max_patch_length $patch_size \
      --vocab_size 256 \
      --quant_range 6 \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 3 \
      --dim_global 16 \
      --dim_local_encoder 8 \
      --dim_local_decoder 8 \
      --n_heads_local_encoder 4 \
      --n_heads_local_decoder 4 \
      --n_heads_global 4 \
      --cross_attn_k 1 \
      --cross_attn_nheads 4 \
      --cross_attn_window_encoder 96 \
      --cross_attn_window_decoder 96 \
      --local_attention_window_len 96 \
      --dropout 0.04 \
      --patching_threshold 0.2 \
      --patching_threshold_add 0.15 \
      --monotonicity 1 \
      --des "patch_size_${patch_size}" \
      --train_epochs 50 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 128 \
      --patching_batch_size 1024 \
      --learning_rate 0.0001 \
      >logs/HyperparamSearch/patch_size_${patch_size}.log 2>&1
done

# Experiment 2: Model Dimension Scaling
echo "Experiment 2: Testing different model dimensions..."
for dim_scale in "small" "medium" "large"
do
    if [ "$dim_scale" = "small" ]; then
        dim_global=8
        dim_local_encoder=4
        dim_local_decoder=4
        n_heads=2
    elif [ "$dim_scale" = "medium" ]; then
        dim_global=16
        dim_local_encoder=8
        dim_local_decoder=8
        n_heads=4
    else # large
        dim_global=32
        dim_local_encoder=16
        dim_local_decoder=16
        n_heads=8
    fi
    
    echo "Running with $dim_scale model dimensions"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_dim_${dim_scale}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --patch_size 8 \
      --max_patch_length 8 \
      --vocab_size 256 \
      --quant_range 6 \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 3 \
      --dim_global $dim_global \
      --dim_local_encoder $dim_local_encoder \
      --dim_local_decoder $dim_local_decoder \
      --n_heads_local_encoder $n_heads \
      --n_heads_local_decoder $n_heads \
      --n_heads_global $n_heads \
      --cross_attn_k 1 \
      --cross_attn_nheads $n_heads \
      --cross_attn_window_encoder 96 \
      --cross_attn_window_decoder 96 \
      --local_attention_window_len 96 \
      --dropout 0.04 \
      --patching_threshold 0.2 \
      --patching_threshold_add 0.15 \
      --monotonicity 1 \
      --des "dim_${dim_scale}" \
      --train_epochs 50 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 128 \
      --patching_batch_size 1024 \
      --learning_rate 0.0001 \
      >logs/HyperparamSearch/dim_${dim_scale}.log 2>&1
done

# Experiment 3: Learning Rate and Scheduler
echo "Experiment 3: Testing different learning rates and schedulers..."
for lr_config in "lr_0001_TST" "lr_0005_TST" "lr_00005_TST" "lr_0001_cosine" "lr_0001_step"
do
    if [[ $lr_config == *"lr_0001"* ]]; then
        lr=0.0001
    elif [[ $lr_config == *"lr_0005"* ]]; then
        lr=0.0005
    elif [[ $lr_config == *"lr_00005"* ]]; then
        lr=0.00005
    fi
    
    if [[ $lr_config == *"TST"* ]]; then
        scheduler="TST"
    elif [[ $lr_config == *"cosine"* ]]; then
        scheduler="cosine"
    elif [[ $lr_config == *"step"* ]]; then
        scheduler="step"
    fi
    
    echo "Running with learning_rate=$lr, scheduler=$scheduler"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${lr_config}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --patch_size 8 \
      --max_patch_length 8 \
      --vocab_size 256 \
      --quant_range 6 \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 3 \
      --dim_global 16 \
      --dim_local_encoder 8 \
      --dim_local_decoder 8 \
      --n_heads_local_encoder 4 \
      --n_heads_local_decoder 4 \
      --n_heads_global 4 \
      --cross_attn_k 1 \
      --cross_attn_nheads 4 \
      --cross_attn_window_encoder 96 \
      --cross_attn_window_decoder 96 \
      --local_attention_window_len 96 \
      --dropout 0.04 \
      --patching_threshold 0.2 \
      --patching_threshold_add 0.15 \
      --monotonicity 1 \
      --des "${lr_config}" \
      --train_epochs 50 \
      --patience 10 \
      --lradj $scheduler \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 128 \
      --patching_batch_size 1024 \
      --learning_rate $lr \
      >logs/HyperparamSearch/${lr_config}.log 2>&1
done

echo "Experiment 4a: Testing different quantization ranges (vocab_size=256)..."
vocab_size=256  # Fixed vocabulary size
for quant_range in 2 4 6 8 10 12
do
    token_config="vocab${vocab_size}_range${quant_range}"
    echo "Running with vocab_size=$vocab_size, quant_range=$quant_range"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${token_config}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --patch_size 8 \
      --max_patch_length 8 \
      --vocab_size $vocab_size \
      --quant_range $quant_range \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 3 \
      --dim_global 16 \
      --dim_local_encoder 8 \
      --dim_local_decoder 8 \
      --n_heads_local_encoder 4 \
      --n_heads_local_decoder 4 \
      --n_heads_global 4 \
      --cross_attn_k 1 \
      --cross_attn_nheads 4 \
      --cross_attn_window_encoder 96 \
      --cross_attn_window_decoder 96 \
      --local_attention_window_len 96 \
      --dropout 0.04 \
      --patching_threshold 0.2 \
      --patching_threshold_add 0.15 \
      --monotonicity 1 \
      --des "quant_range_${quant_range}" \
      --train_epochs 50 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 128 \
      --patching_batch_size 1024 \
      --learning_rate 0.0001 \
      >logs/HyperparamSearch/quant_range_${quant_range}.log 2>&1
done

# Experiment 5: Layer Depth Variations
echo "Experiment 5: Testing different layer configurations..."
for layer_config in "shallow" "medium" "deep"
do
    if [ "$layer_config" = "shallow" ]; then
        n_layers_local_encoder=1
        n_layers_local_decoder=1
        n_layers_global=2
    elif [ "$layer_config" = "medium" ]; then
        n_layers_local_encoder=2
        n_layers_local_decoder=2
        n_layers_global=3
    else # deep
        n_layers_local_encoder=3
        n_layers_local_decoder=3
        n_layers_global=4
    fi
    
    echo "Running with $layer_config layer configuration"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_layers_${layer_config}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --patch_size 8 \
      --max_patch_length 8 \
      --vocab_size 256 \
      --quant_range 6 \
      --n_layers_local_encoder $n_layers_local_encoder \
      --n_layers_local_decoder $n_layers_local_decoder \
      --n_layers_global $n_layers_global \
      --dim_global 16 \
      --dim_local_encoder 8 \
      --dim_local_decoder 8 \
      --n_heads_local_encoder 4 \
      --n_heads_local_decoder 4 \
      --n_heads_global 4 \
      --cross_attn_k 1 \
      --cross_attn_nheads 4 \
      --cross_attn_window_encoder 96 \
      --cross_attn_window_decoder 96 \
      --local_attention_window_len 96 \
      --dropout 0.04 \
      --patching_threshold 0.2 \
      --patching_threshold_add 0.15 \
      --monotonicity 1 \
      --des "layers_${layer_config}" \
      --train_epochs 50 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size 128 \
      --patching_batch_size 1024 \
      --learning_rate 0.0001 \
      >logs/HyperparamSearch/layers_${layer_config}.log 2>&1
done

# Experiment 6: Batch Size and Dropout
echo "Experiment 6: Testing different batch sizes and dropout rates..."
for batch_config in "batch64_drop002" "batch128_drop004" "batch256_drop008"
do
    if [[ $batch_config == *"batch64"* ]]; then
        batch_size=64
    elif [[ $batch_config == *"batch128"* ]]; then
        batch_size=128
    elif [[ $batch_config == *"batch256"* ]]; then
        batch_size=256
    fi
    
    if [[ $batch_config == *"drop002"* ]]; then
        dropout=0.02
    elif [[ $batch_config == *"drop004"* ]]; then
        dropout=0.04
    elif [[ $batch_config == *"drop008"* ]]; then
        dropout=0.08
    fi
    
    echo "Running with batch_size=$batch_size, dropout=$dropout"
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${batch_config}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --patch_size 8 \
      --max_patch_length 8 \
      --vocab_size 256 \
      --quant_range 6 \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 3 \
      --dim_global 16 \
      --dim_local_encoder 8 \
      --dim_local_decoder 8 \
      --n_heads_local_encoder 4 \
      --n_heads_local_decoder 4 \
      --n_heads_global 4 \
      --cross_attn_k 1 \
      --cross_attn_nheads 4 \
      --cross_attn_window_encoder 96 \
      --cross_attn_window_decoder 96 \
      --local_attention_window_len 96 \
      --dropout $dropout \
      --patching_threshold 0.2 \
      --patching_threshold_add 0.15 \
      --monotonicity 1 \
      --des "${batch_config}" \
      --train_epochs 50 \
      --patience 10 \
      --lradj 'TST' \
      --pct_start 0.4 \
      --itr 1 \
      --batch_size $batch_size \
      --patching_batch_size 1024 \
      --learning_rate 0.0001 \
      >logs/HyperparamSearch/${batch_config}.log 2>&1
done

echo "=== Hyperparameter search completed! ==="
echo "Results saved in logs/HyperparamSearch/"
echo ""
echo "To analyze results, run:"
echo "grep -r 'test mse' logs/HyperparamSearch/ | sort -k3 -n"