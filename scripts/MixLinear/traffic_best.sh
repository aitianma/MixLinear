if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MixLinear

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=Traffic
data_name=custom
seq_len=720

alpha=0.5
pred_len=96
batch_size=64
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
    --enc_in 862 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 19\
    --gpu 7 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log & 


alpha=0.01

pred_len=192
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
    --enc_in 862 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 15 \
    --gpu 7 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log &

alpha=0.5


pred_len=336
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
    --enc_in 862 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 19\
    --gpu 7 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log &

alpha=0.9
pred_len=720
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
    --enc_in 862 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --lpf 19 \
    --gpu 7 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log &


