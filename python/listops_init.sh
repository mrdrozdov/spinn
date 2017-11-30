# RNN runs
SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 300 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00116904131098 --training_data_path spinn/data/listops/train_d10c.tsv --batch_size 32 --learning_rate_decay_per_10k_steps 0.655824165202 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.45798998353e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 32 --seq_length 300 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type RNN --word_embedding_dim 64 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10c.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10c_rnn" bash ../scripts/sbatch_submit.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 300 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00116904131098 --training_data_path spinn/data/listops/train_d10f.tsv --batch_size 32 --learning_rate_decay_per_10k_steps 0.655824165202 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.45798998353e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 32 --seq_length 300 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type RNN --word_embedding_dim 64 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10f_rnn" bash ../scripts/sbatch_submit.sh


# Chart-parsing w/ ST-Gumbel runs
SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 300 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.0036953018599 --training_data_path spinn/data/listops/train_d10c.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.424151296811 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.61672684966e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 64 --seq_length 100 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type Maillard --word_embedding_dim 64 --use_internal_parser  --ckpt_path /scratch/nn1119/maillard/ --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10c.tsv --allow_eval_cropping --allow_cropping --st_gumbel --experiment_name listops_d10c_mail" bash ../scripts/sbatch_submit.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 300 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.0036953018599 --training_data_path spinn/data/listops/train_d10f.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.424151296811 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.61672684966e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 64 --seq_length 100 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type Maillard --word_embedding_dim 64 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --st_gumbel --experiment_name listops_d10f_mail" bash ../scripts/sbatch_submit.sh