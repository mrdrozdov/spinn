#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

DATA=arithmetic
DEVAL=../../spinn/python/spinn/data/arithmetic/simple5_1k.tsv
DTRAIN=../../spinn/python/spinn/data/arithmetic/simple5_10k.tsv
LRs=(0.0001)
RWs=(1.0 0.1 0.001)
GPU=0
STEPS=100000
OPT=RMSprop

export PYTHONPATH=$PYTHONPATH:./python:../python:spinn/python
cd ~/Developer/spinn/checkpoints

for LR in "${LRs[@]}"
do
    for RW in "${RWs[@]}"
    do
        python -m spinn.models.fat_classifier \
            --batch_size 40 \
            --data_type $DATA \
            --eval_data_path $DEVAL \
            --log_path ../logs \
            --ckpt_path ../logs \
            --model_dim 50 \
            --model_type RLSPINN \
            --seq_length 50 \
            --training_data_path $DTRAIN \
            --word_embedding_dim 40 \
            --eval_interval_steps 100 \
            --tracking_lstm_hidden_dim 44 \
            --rl_weight $RW \
            --transition_weight 0.6 \
            --statistics_interval_steps 100 \
            --semantic_classifier_keep_rate 0.90 \
            --use_internal_parser \
            --l2_lambda 2.75e-05 \
            --learning_rate $LR \
            --gpu $GPU \
            --optimizer_type $OPT \
            --training_steps $STEPS \
            --rl_baseline greedy \
            --experiment_name $DATA-lr.$LR-rlw.$RW-greedy
    done
done
