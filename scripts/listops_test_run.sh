#!/bin/bash

LOG_DIR='.'

# Test run for SPINN listops. Small model, quick runs, just to make sure
# nothing catches on fire before you commit to a bigger model.

# Dimensions are intentionally set as prime numbers to catch issues.
MODEL="spinn.models.fat_classifier"
FLAGS="  --eval_seq_length 3000
  --metrics_path /home/sb6065/logs/spinn-runs
  --eval_interval_steps 101
  --data_type listops
  --log_path $LOG_DIR
  --embedding_keep_rate 1.0
  --learning_rate 0.00641719241862
  --training_data_path spinn/data/listops/train_d20a.tsv
  --batch_size 19
  --learning_rate_decay_per_10k_steps 0.553467172659
  --mlp_dim 23
  --nouse_tracking_in_composition  
  --nolateral_tracking
  --encode pass
  --semantic_classifier_keep_rate 1.0
  --model_type SPINN
  --model_dim 34
  --seq_length 83
  --statistics_interval_steps 40
  --num_mlp_layers 2
  --l2_lambda 3.65878847025e-05
  --word_embedding_dim 34
  --use_internal_parser
  --ckpt_path $LOG_DIR
  --transition_weight 0.1
  --eval_data_path spinn/data/listops/test_d20a.tsv
  --experiment_name listops_test_run;"

# Run from the 'python' directory.
SPINNMODEL=$MODEL SPINN_FLAGS=$FLAGS bash ../scripts/train_spinn_cpu_only.sbatch
