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
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

random_seed=2025
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
      --enc_in 7 \
      --vocab_size 256 \
      --quant_range 2 \
      --n_layers_local_encoder 2 \
      --n_layers_local_decoder 2 \
      --n_layers_global 2 \
      --dim_global 32 \
      --dim_local_encoder 16 \
      --dim_local_decoder 16 \
      --cross_attn_k 1 \
      --n_heads_local_encoder 2 \
      --n_heads_local_decoder 2 \
      --n_heads_global 2 \
      --cross_attn_nheads 2 \
      --cross_attn_window_encoder 96\
      --cross_attn_window_decoder 96\
      --local_attention_window_len 96\
      --dropout 0.05\
      --multiple_of 128\
      --patch_size 16\
      --max_patch_length 16\
      --patching_threshold 1\
      --patching_threshold_add 0.4\
      --monotonicity 1\
      --des 'Exp' \
      --train_epochs 140\
      --patience 30\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 \
      --batch_size 256 \
      --patching_batch_size 1792 \
      --learning_rate 0.00006 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done