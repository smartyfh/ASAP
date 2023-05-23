#!/bin/bash

for layer in 2 4 6 8 10 12
do
    for gamma in 0 0.5 1
    do
        CUDA_VISIBLE_DEVICES=0 python main.py \
            --data=sgd \
            --seed=42 \
            --gpu=0 \
            --batch_size=16 \
            --epoch_num=4 \
            --max_seq_len=64 \
            --max_dial_len=16 \
            --eval_steps=100 \
            --lr=1e-5 \
            --gamma=${gamma} \
            --tf_heads=12 \
            --tf_layers=2 \
            --hp_heads=12 \
            --hp_layers=${layer} \
            --rewrite_data 
    done
done
