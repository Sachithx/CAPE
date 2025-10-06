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
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
batch_size=128
enc_in=7
vocab_size=256
random_seed=2025
quant_range=3
max_patch_length=12
learning_rate=0.01
dropout=0.02
train_epochs=10
patience=7

for batch_size in 128 32 256 64 512  
do
    for patching_threshold in 3 4 3.5  
    do
        for patching_threshold_add in 0.1 0.3 0.5 0.2 0.8
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
            --enc_in $enc_in \
            --vocab_size $vocab_size \
            --quant_range $quant_range \
            --n_layers_local_encoder 1 \
            --n_layers_local_decoder 1 \
            --n_layers_global 1 \
            --dim_global 16 \
            --dim_local_encoder 16 \
            --dim_local_decoder 16 \
            --cross_attn_k 1 \
            --n_heads_local_encoder 2 \
            --n_heads_local_decoder 2 \
            --n_heads_global 2 \
            --cross_attn_nheads 2 \
            --dropout $dropout \
            --multiple_of 128 \
            --max_patch_length $max_patch_length \
            --patching_threshold $patching_threshold \
            --patching_threshold_add $patching_threshold_add \
            --monotonicity 0 \
            --des Exp \
            --train_epochs $train_epochs \
            --patience $patience \
            --lradj type3 \
            --pct_start 0.3 \
            --itr 1 \
            --batch_size $batch_size \
            --patching_batch_size $((batch_size * enc_in)) \
            --learning_rate $learning_rate \
            > "logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
        done
    done
done