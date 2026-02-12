if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

#model_name=SparseTSF
model_name=MixLinear

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
alpha=0.99

lpf=30
seq_len=720
for lpf in 2 15
do
for alpha in   0.01 0.99  
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
    --period_len 24 \
    --enc_in 7 \
    --train_epochs 30 \
    --patience 5 \
    --alpha $alpha \
    --gpu 3 \
    --lpf $lpf \
    --itr 1 --batch_size 256 --learning_rate 0.02 > logs/${model_name}_${data_name}_${pred_len}_${lpf}_${alpha}_lpf.log  &
done
done
done

