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
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
batch_size=256
enc_in=7
vocab_size=256
random_seed=2025
quant_range=3
learning_rate=0.001
train_epochs=30
patience=5

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --entropy_model_checkpoint_dir $entropy_model_checkpoint_dir \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
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
      --dim_global 32 \
      --dim_local_encoder 32 \
      --dim_local_decoder 32 \
      --cross_attn_k 1 \
      --n_heads_local_encoder 8 \
      --n_heads_local_decoder 8 \
      --n_heads_global 8 \
      --cross_attn_nheads 8 \
      --dropout 0.5 \
      --multiple_of 256 \
      --patch_size 96 \
      --max_patch_length $seq_len \
      --patching_threshold 0.3 \
      --patching_threshold_add 0.05 \
      --monotonicity 1 \
      --des 'Exp' \
      --train_epochs $train_epochs \
      --patience $patience \
      --lradj 'type3'\
      --pct_start 0.3 \
      --itr 1 \
      --batch_size $batch_size \
      --patching_batch_size $((batch_size * enc_in)) \
      --learning_rate $learning_rate \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done