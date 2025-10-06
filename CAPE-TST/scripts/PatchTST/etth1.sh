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
data_path_name=ETTh1.csv
model_id_name=ETTh1  
data_name=ETTh1

random_seed=2025

# Function to get random value from space-separated list
get_random() {
    local values="$1"
    local array=($values)
    local random_index=$((RANDOM % ${#array[@]}))
    echo "${array[$random_index]}"
}

# Define parameter ranges
pred_lens="96 192 336 720"
n_layers_options="1 2 3 4 6 8"
dim_options="8 16 32 64 128"
n_heads_options="1 2 4 8 16"
patch_size_options="4 8 12 16 24 32"
quant_range_options="1 2 3 4 8"
learning_rates="0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1"
batch_sizes="32 64 128 256 512"
dropout_rates="0.0 0.05 0.1 0.15 0.2 0.3"
patching_thresholds="0.1 0.2 0.25 0.3 0.35 0.4 0.5"
patching_threshold_adds="0.05 0.1 0.15 0.2 0.25"
weight_decay_options="1e-3"
grad_clip_options="0.5"
pct_start_options="0.1 0.2 0.3 0.4 0.5"
train_epochs_options="50 80 120 180 240"
patience_options="15 25 40 60 80"
cross_attn_k_options="1 2 4"
window_sizes="48 96 144 192"
multiple_of_options="16 32 64 128"

num_experiments=50
echo "Running $num_experiments diverse random experiments..."

for i in $(seq 1 $num_experiments); do
    echo "=== Experiment $i/$num_experiments ==="

    # Use time + PID + iteration to reseed randomness
    RANDOM=$(( $(date +%s) + $$ + i ))

    pred_len=$(get_random "$pred_lens")

    n_layers_local_encoder=$(get_random "$n_layers_options")
    n_layers_local_decoder=$(get_random "$n_layers_options")
    n_layers_global=$(get_random "$n_layers_options")

    dim_global=$(get_random "$dim_options")
    dim_local_encoder=$(get_random "$dim_options")
    dim_local_decoder=$(get_random "$dim_options")

    n_heads_local_encoder=$(get_random "$n_heads_options")
    n_heads_local_decoder=$(get_random "$n_heads_options")
    n_heads_global=$(get_random "$n_heads_options")
    cross_attn_nheads=$(get_random "$n_heads_options")

    patch_size=$(get_random "$patch_size_options")
    max_patch_length=$patch_size
    quant_range=$(get_random "$quant_range_options")

    learning_rate=$(get_random "$learning_rates")
    batch_size=$(get_random "$batch_sizes")
    dropout=$(get_random "$dropout_rates")

    patching_batch_size=$((batch_size * 7))

    patching_threshold=$(get_random "$patching_thresholds")
    patching_threshold_add=$(get_random "$patching_threshold_adds")

    weight_decay=$(get_random "$weight_decay_options")
    grad_clip_norm=$(get_random "$grad_clip_options")

    pct_start=$(get_random "$pct_start_options")
    train_epochs=$(get_random "$train_epochs_options")
    patience=$(get_random "$patience_options")

    cross_attn_k=$(get_random "$cross_attn_k_options")
    window_size=$(get_random "$window_sizes")

    monotonicity=$((RANDOM % 2))
    multiple_of=$(get_random "$multiple_of_options")

    experiment_id="${i}_$(date +%s)"

    echo "Config: pred_len=$pred_len, layers=[$n_layers_local_encoder,$n_layers_local_decoder,$n_layers_global]"
    echo "        dims=[$dim_local_encoder,$dim_local_decoder,$dim_global], heads=[$n_heads_local_encoder,$n_heads_local_decoder,$n_heads_global]"
    echo "        patch_size=$patch_size, lr=$learning_rate, bs=$batch_size, dropout=$dropout"
    echo "        threshold=$patching_threshold, weight_decay=$weight_decay"

    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${seq_len}_${pred_len}_exp${experiment_id} \
      --model_id_name $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --vocab_size 256 \
      --quant_range $quant_range \
      --n_layers_local_encoder $n_layers_local_encoder \
      --n_layers_local_decoder $n_layers_local_decoder \
      --n_layers_global $n_layers_global \
      --dim_global $dim_global \
      --dim_local_encoder $dim_local_encoder \
      --dim_local_decoder $dim_local_encoder \
      --cross_attn_k $cross_attn_k \
      --n_heads_local_encoder $n_heads_local_encoder \
      --n_heads_local_decoder $n_heads_local_decoder \
      --n_heads_global $n_heads_global \
      --cross_attn_nheads $cross_attn_nheads \
      --cross_attn_window_encoder $window_size \
      --cross_attn_window_decoder $window_size \
      --local_attention_window_len $window_size \
      --dropout $dropout \
      --multiple_of $multiple_of \
      --patch_size $patch_size \
      --max_patch_length $max_patch_length \
      --patching_threshold $patching_threshold \
      --patching_threshold_add $patching_threshold_add \
      --monotonicity $monotonicity \
      --des 'Exp' \
      --train_epochs $train_epochs \
      --patience $patience \
      --lradj 'TST' \
      --pct_start $pct_start \
      --itr 1 \
      --batch_size $batch_size \
      --patching_batch_size $patching_batch_size \
      --learning_rate $learning_rate \
      >logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}_exp${experiment_id}.log 2>&1

    echo "Completed experiment $i"
    echo "Log: ${model_name}_${model_id_name}_${seq_len}_${pred_len}_exp${experiment_id}.log"
    echo "----------------------------------------"

    sleep 2
done

echo "All $num_experiments diverse experiments completed!"
