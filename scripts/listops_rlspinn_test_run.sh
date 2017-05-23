#!/bin/bash

LOG_DIR='.'

# Test run for RL-SPINN listops. Small model, quick runs, just to make sure
# nothing catches on fire before you commit to a bigger model.

# Dimensions are intentionally set as prime numbers to catch issues.
MODEL="spinn.models.reinforce"
FLAGS="
--eval_seq_length 1800
--norl_catalan 
--sample_interval_steps 100
--eval_interval_steps 100
--data_type listops
--log_path .
--embedding_keep_rate 1.0
--learning_rate 0.000209173541291
--training_data_path spinn/data/listops/train_d20.tsv
--rl_weight 0.095492504495
--batch_size 64
--learning_rate_decay_per_10k_steps 0.514994481626
--mlp_dim 16
--nouse_tracking_in_composition 
--rl_baseline greedy
--norl_wake_sleep 
--nolateral_tracking 
--rl_valid 
--encode pass
--semantic_classifier_keep_rate 1.0
--rl_reward xent
--model_type RLSPINN
--model_dim 8
--num_samples 5
--seq_length 1800
--statistics_interval_steps 20
--num_mlp_layers 2
--rl_wake_sleep 
--nopredict_use_cell 
--l2_lambda 1.99440298507e-05
--word_embedding_dim 8
--use_internal_parser 
--ckpt_path .
--transition_weight 1
--eval_data_path spinn/data/listops/test_d20.tsv
--experiment_name listops_rlspinn_test_run"

# Run from the 'python' directory.
SPINNMODEL=$MODEL SPINN_FLAGS=$FLAGS ../scripts/train_spinn_cpu_only.sbatch
