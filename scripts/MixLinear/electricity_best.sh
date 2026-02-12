if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MixLinear

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
seq_len=720

lpf=19
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
    --enc_in 321 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 2 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log & 


alpha=0.99
pred_len=192
lpf=15
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
    --enc_in 321 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --gpu 2 \
    --lpf $lpf \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log &


alpha=0.99
pred_len=336
lpf=15
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
    --enc_in 321 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log &


alpha=0.5
pred_len=720
lpf=15
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
    --enc_in 321 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --lpf $lpf \
    --gpu 3 \
    --itr 1 --batch_size $batch_size --learning_rate 0.03 > logs/${model_name}_${model_id_name}_${pred_len}_${batch_size}_${alpha}_best.log &


