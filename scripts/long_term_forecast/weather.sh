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
dataset=Weather
seq_lens=(512 512 336 336)
pred_lens=(96 192 336 720)
learning_rates=(0.00085 0.001 0.0007 0.003)
batches=(32 32 1024 1024)
wavelets=(db3 db3 db3 db2)
levels=(1 1 1 1)
tfactors=(3 3 7 7)
dfactors=(7 7 7 5)
cfactors=(3 6 8 3)
epochs=(60 60 60 60)
dropouts=(0.4 0.4 0.4 0.4)
embedding_dropouts=(0.1 0.0 0.4 0.2)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 128 128)
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
		--train_epochs ${epochs[$i]} > $log_file
done