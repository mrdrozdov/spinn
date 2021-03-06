export PYTHONPATH=./python

python -m spinn.models.supervised_classifier \
--allow_eval_cropping \
--batch_size 128 \
--ckpt_interval_steps 500 \
--ckpt_path ./logs \
--ckpt_step 500 \
--data_type nli  \
--embedding_data_path ~/data/glove/glove.6B.300d.txt \
--embedding_keep_rate 0.966213295977 \
--encode_bidirectional \
--eval_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
--eval_interval_steps 500 \
--eval_seq_length 35 \
--experiment_name demo \
--l2_lambda 1.857572718e-07 \
--learning_rate 0.000438237201979 \
--log_path ./logs \
--mlp_dim 512 \
--model_dim 300 \
--model_type Maillard \
--nolateral_tracking \
--nouse_tracking_in_composition \
--num_mlp_layers 1 \
--num_samples 1 \
--semantic_classifier_keep_rate 0.900692453701 \
--seq_length 35 \
--statistics_interval_steps 100 \
--training_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
--transition_weight 1.0 \
--use_internal_parser \
--word_embedding_dim 300
