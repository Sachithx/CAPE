if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PatchTST

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021
for pred_len in 96 # 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --vocab_size 256 \
      --quant_range 6 \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 3 \
      --dim_global 32 \
      --dim_local_encoder 8 \
      --dim_local_decoder 8 \
      --cross_attn_k 1 \
      --n_heads_local_encoder 4 \
      --n_heads_local_decoder 4 \
      --n_heads_global 4 \
      --cross_attn_nheads 4 \
      --cross_attn_window_encoder 96\
      --cross_attn_window_decoder 96\
      --local_attention_window_len 96\
      --dropout 0.04\
      --patch_size 8\
      --max_patch_length 8\
      --patching_threshold 0.2\
      --patching_threshold_add 0.15\
      --monotonicity 1\
      --des 'Exp' \
      --train_epochs 100\
      --patience 20\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 \
      --batch_size 128 \
      --patching_batch_size 512 \
      --learning_rate 0.0001 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done