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
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

random_seed=2025
for pred_len in 720 336 192 96
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
      --enc_in 321 \
      --vocab_size 256 \
      --quant_range 4 \
      --n_layers_local_encoder 4 \
      --n_layers_local_decoder 4 \
      --n_layers_global 8 \
      --dim_global 128 \
      --dim_local_encoder 32 \
      --dim_local_decoder 32 \
      --cross_attn_k 1 \
      --n_heads_local_encoder 8 \
      --n_heads_local_decoder 8 \
      --n_heads_global 16 \
      --cross_attn_nheads 8 \
      --cross_attn_window_encoder 96\
      --cross_attn_window_decoder 96\
      --local_attention_window_len 96\
      --dropout 0.2\
      --multiple_of 128\
      --patch_size 8\
      --max_patch_length 8\
      --patching_threshold 0.3\
      --patching_threshold_add 0.05\
      --monotonicity 1\
      --des 'Exp' \
      --train_epochs 50\
      --patience 10\
      --lradj 'TST'\
      --pct_start 0.4\
      --itr 1 \
      --batch_size 32 \
      --patching_batch_size 10272 \
      --learning_rate 0.0005 \
      >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done