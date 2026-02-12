if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

model_name=MixLinear

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
alpha=0.01

seq_len=720
for lpf in   1
do
for alpha in  0.01 0.5 0.99
do
for pred_len in 96 192 336 720
do
  ~/bin/python3 -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 4 \
    --enc_in 21 \
    --train_epochs 30 \
    --patience 10 \
    --alpha $alpha \
    --gpu 4 \
    --itr 1 --batch_size 64 --learning_rate 0.02 > logs/${model_name}_${model_id_name}_${pred_len}_${lpf}_${alpha}.log  &
done
done
done
