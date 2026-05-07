#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Create WaveMixerNet logs directory if it doesn't exist
if [ ! -d "./logs/WaveMixerNet" ]; then
	mkdir ./logs/WaveMixerNet
fi

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WaveMixerNet

# Datasets and prediction lengths
dataset=ETTh2
seq_lens=(512 512 336 512)
pred_lens=(96 192 336 720)
learning_rates=(0.00015 0.0009 0.000814275 0.00075)
batches=(1024 1024 512 256)
wavelets=(db1 db1 sym3 db2)
levels=(1 1 1 1)
tfactors=(5 3 9 5)
dfactors=(5 8 6 5)
cfactors=(6 2 3 9)
epochs=(30 30 30 30)
dropouts=(0.0 0.0 0.4 0.4)
embedding_dropouts=(0.1 0.0 0.4 0.0)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 128)
patiences=(12 12 12 12)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	log_file="logs/${model_name}/long_term_forecast_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--cfactor ${cfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done