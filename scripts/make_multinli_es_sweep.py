# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "0811_stnd"
SWEEP_RUNS = 8

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
BOOL = "BOOL"
CHOICE = "CHOICE"
MUL = "MUL" # multiple of 100

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "/home/nn1119/data/multinli_0.9_snli_1.0_train_combined.jsonl", "")
gflags.DEFINE_string("eval_data_path", "/home/nn1119/data/multinli_0.9_dev_matched.jsonl", "")
gflags.DEFINE_string("embedding_data_path", "/home/nn1119/data/glove.840B.300d.txt", "")
gflags.DEFINE_string("log_path", "/scratch/nn1119/spinn/nli/sweep", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "nli",
    "model_type":      "SPINN",
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "embedding_data_path": FLAGS.embedding_data_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "300",
    "model_dim":   "600",
    "seq_length":   "80",
    "eval_seq_length":  "810",
    "eval_interval_steps": "100",
    "sample_interval_steps": "100",
    "statistics_interval_steps": "100",
    "ckpt_step": "100",
    "ckpt_interval_steps": "100",
    "batch_size":  "32",
    "encode": "gru",
    "encode_bidirectional": "", 
    "num_mlp_layers": "1",
    "mlp_dim": "1024",
    "nocomposition_ln": "",
    "write_proto_to_log": "",
    "allow_eval_cropping": "",
    "use_internal_parser": "",
    "transition_weight": "1.0",
    "num_mlp_layers": "1",
    "num_samples": "1",
    "evolution": "",
    "embedding_keep_rate": "1.0"
    "eval_sample_size": "0.3",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "semantic_classifier_keep_rate": ("skr",    LIN, 0.8, 1.0),
    "l2_lambda":          ("l2", EXP, 1e-8, 2e-5),
    "learning_rate_decay_per_10k_steps": ("dc", LIN, 0.4, 1.0),
    "learning_rate": ("lr", EXP, 0.00005, 0.005),
    "tracking_lstm_hidden_dim": ("tdim", LIN, 20, 40),
    "es_num_episodes" : ("eps", LIN, 4, 5),
    "es_num_roots" : ("roots", LIN, 1, 4),
    "es_episode_length" : ("lng", MUL, 150, 550),
    "es_sigma": ("sig", EXP, 0.01, 0.08),
}


sweep_name = "sweep_" + NAME + "_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(SWEEP_RUNS)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

for run_id in range(SWEEP_RUNS):
    params = {}
    name = sweep_name + "_" + str(run_id)

    params.update(FIXED_PARAMETERS)
    # Any param appearing in both sets will be overwritten by the sweep value.

    for param in SWEEP_PARAMETERS:
        config = SWEEP_PARAMETERS[param]
        t = config[1]
        mn = config[2]
        mx = config[3]

        r = random.uniform(0, 1)
        if t == EXP:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = np.exp(lmn + (lmx - lmn) * r)
        elif t == BOOL:
            sample = r > 0.5
        elif t==SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        elif t==CHOICE:
            sample = random.choice(mn)
        elif t==MUL:
            x = mn + (mx - mn) * r
            sample =  x + 100 - x % 100
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
            params[param] = sample
        elif isinstance(mn, float):
            val_disp = "%.2g" % sample
            params[param] = sample
        elif t==BOOL:
            val_disp = str(int(sample))
            if not sample:
                params['no' + param] = ''
            else:
                params[param] = ''
        else:
            val_disp = sample
            params[param] = sample
        name += "-" + config[0] + val_disp

    flags = ""
    for param in params:
        value = params[param]
        flags += " --" + param + " " + str(value)
        if param == "es_num_roots":
            root = value
        elif param == "es_num_episodes":
            eps = value
    num_cores = root * eps

    flags += " --experiment_name " + name
    if NYU_NON_PBS:
        print "cd spinn/python; python2.7 -m spinn.models.es_classifier " + flags
    else:
        if num_cores <= 10:
            print "SPINNMODEL=\"spinn.models.es_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_es_cpu_only.sh"
        elif num_cores <= 15:
            print "SPINNMODEL=\"spinn.models.es_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_es_cpu_only_15.sh"
        else:
            print "SPINNMODEL=\"spinn.models.es_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_es_cpu_only_20.sh"
    print
