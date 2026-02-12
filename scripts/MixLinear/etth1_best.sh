if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MixLinear

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
seq_len=720

lpf=1

batch_size=256
pred_len=96
alpha=0.95
lpf=1
python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --gpu 3 \
    --lpf $lpf \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${data_name}_${pred_len}_${batch_size}_${lpf}_${alpha}_best.log  &

lpf=5
pred_len=192
alpha=0.95
python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${data_name}_${pred_len}_${batch_size}_${lpf}_${alpha}_best.log  &

pred_len=336
alpha=0.99
lpf=2
python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${data_name}_${pred_len}_${batch_size}_${lpf}_${alpha}_best.log  &

pred_len=720
alpha=0.99
lpf=19
python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${data_name}_${pred_len}_${batch_size}_${lpf}_${alpha}_best.log  &

