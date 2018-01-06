# RNN runs
SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 3000 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00116904131098 --training_data_path spinn/data/listops/train_d10c.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.655824165202 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.45798998353e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 64 --seq_length 3000 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type RNN --word_embedding_dim 64 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --transition_weight 1.0 --eval_data_path spinn/data/listops/test_d10c.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10c_rnn_d64_sl3000" bash ../scripts/sbatch_submit_cpu_only.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 1000 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00116904131098 --training_data_path spinn/data/listops/train_d10f.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.655824165202 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.45798998353e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 32 --seq_length 1000 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type RNN --word_embedding_dim 32 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10f_rnn_d32_sl1000" bash ../scripts/sbatch_submit_cpu_only.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 3000 --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00116904131098 --training_data_path spinn/data/listops/train_d10f.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.655824165202 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.45798998353e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 64 --seq_length 300 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type RNN --word_embedding_dim 64 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10f_rnn_d64_sl3000" bash ../scripts/sbatch_submit_cpu_only.sh

# SPINN supervised
SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 1000 --training_data_path spinn/data/listops/train_d10c.tsv --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00485446189576 --batch_size 64 --learning_rate_decay_per_10k_steps 0.353826263567 --mlp_dim 16 --statistics_interval_steps 100 --l2_lambda 3.17258908262e-06 --nolateral_tracking  --semantic_classifier_keep_rate 1.0 --use_internal_parser  --model_dim 64 --seq_length 100 --num_mlp_layers 2 --model_type SPINN --word_embedding_dim 64 --encode pass --ckpt_path /scratch/nn1119/maillard/ --nouse_tracking_in_composition  --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10c.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10c_spinn_d64_esl1000" bash ../scripts/sbatch_submit_cpu_only.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 3000 --training_data_path spinn/data/listops/train_d10c.tsv --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00485446189576 --batch_size 64 --learning_rate_decay_per_10k_steps 0.353826263567 --mlp_dim 16 --statistics_interval_steps 100 --l2_lambda 3.17258908262e-06 --nolateral_tracking  --semantic_classifier_keep_rate 1.0 --use_internal_parser  --model_dim 64 --seq_length 100 --num_mlp_layers 2 --model_type SPINN --word_embedding_dim 64 --encode pass --ckpt_path /scratch/nn1119/maillard/ --nouse_tracking_in_composition  --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10c.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10c_spinn_d64_esl3000" bash ../scripts/sbatch_submit_cpu_only.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 1000 --training_data_path spinn/data/listops/train_d10f.tsv --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00485446189576 --batch_size 64 --learning_rate_decay_per_10k_steps 0.353826263567 --mlp_dim 16 --statistics_interval_steps 100 --l2_lambda 3.17258908262e-06 --nolateral_tracking  --semantic_classifier_keep_rate 1.0 --use_internal_parser  --model_dim 64 --seq_length 100 --num_mlp_layers 2 --model_type SPINN --word_embedding_dim 64 --encode pass --ckpt_path /scratch/nn1119/maillard/ --nouse_tracking_in_composition  --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10c_spinn_d64_esl1000" bash ../scripts/sbatch_submit_cpu_only.sh

SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 3000 --training_data_path spinn/data/listops/train_d10f.tsv --eval_interval_steps 100 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.00485446189576 --batch_size 64 --learning_rate_decay_per_10k_steps 0.353826263567 --mlp_dim 16 --statistics_interval_steps 100 --l2_lambda 3.17258908262e-06 --nolateral_tracking  --semantic_classifier_keep_rate 1.0 --use_internal_parser  --model_dim 64 --seq_length 100 --num_mlp_layers 2 --model_type SPINN --word_embedding_dim 64 --encode pass --ckpt_path /scratch/nn1119/maillard/ --nouse_tracking_in_composition  --transition_weight 0.1 --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --experiment_name listops_d10c_spinn_d64_esl3000" bash ../scripts/sbatch_submit_cpu_only.sh

# Chart-parsing w/ ST-Gumbel runs
#SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 300 --eval_interval_steps 50 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.0036953018599 --training_data_path spinn/data/listops/train_d10c.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.424151296811 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.61672684966e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 64 --seq_length 50 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type Maillard --word_embedding_dim 64 --use_internal_parser  --ckpt_path /scratch/nn1119/maillard/  --eval_data_path spinn/data/listops/test_d10c.tsv --allow_eval_cropping --allow_cropping --st_gumbel --experiment_name listops_d10c_mail" bash ../scripts/sbatch_submit_once.sh

#SPINNMODEL="spinn.models.supervised_classifier" SPINN_FLAGS=" --eval_seq_length 300 --eval_interval_steps 50 --data_type listops --embedding_keep_rate 1.0 --learning_rate 0.0036953018599 --training_data_path spinn/data/listops/train_d10f.tsv --batch_size 64 --learning_rate_decay_per_10k_steps 0.424151296811 --mlp_dim 16 --nouse_tracking_in_composition  --l2_lambda 1.61672684966e-06 --nolateral_tracking  --encode pass --semantic_classifier_keep_rate 1.0 --model_dim 64 --seq_length 50 --statistics_interval_steps 100 --num_mlp_layers 2 --model_type Maillard --word_embedding_dim 64 --use_internal_parser --ckpt_path /scratch/nn1119/maillard/ --eval_data_path spinn/data/listops/test_d10f.tsv --allow_eval_cropping --allow_cropping --st_gumbel --experiment_name listops_d10f_mail" bash ../scripts/sbatch_submit_once.sh
