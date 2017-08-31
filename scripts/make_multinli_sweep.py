# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "big_enc"
SWEEP_RUNS = 5

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
BOOL = "BOOL"
CHOICE = "CHOICE"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "/home/sb6065/multinli_0.9/multinli_0.9_snli_1.0_train_combined.jsonl", "")
gflags.DEFINE_string("eval_data_path", "/home/sb6065/multinli_0.9/multinli_0.9_dev_matched.jsonl", "")
gflags.DEFINE_string("embedding_data_path", "/home/sb6065/glove/glove.840B.300d.txt", "")
gflags.DEFINE_string("log_path", "/scratch/sb6065/logs/spinn", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "nli",
    "model_type":      "ChoiPyramid",
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "embedding_data_path": FLAGS.embedding_data_path,
    "log_path": FLAGS.log_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "300",
    "model_dim":   "1200",
    "seq_length":   "80",
    "eval_seq_length":  "810",
    "eval_interval_steps": "1000",
    "sample_interval_steps": "1000",
    "statistics_interval_steps": "100",
    "batch_size":  "32",
    "encode": "gru",
    "encode_bidirectional": "", 
    "num_mlp_layers": "1",
    "mlp_dim": "1024",
    #"nocomposition_ln": "",
    "embedding_keep_rate": "1.0",
    "pyramid_trainable_temperature": "",
    "learning_rate_decay_per_10k_steps": "1.0",
    "pyramid_temperature_decay_per_10k_steps": "1.0", 
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "semantic_classifier_keep_rate": ("skr", LIN, 0.5, 1.0),
    "l2_lambda":          ("l2l", EXP, 3e-9, 3e-6),
    "learning_rate": ("lr", EXP, 0.00003, 0.001),
}


sweep_name = "sweep_" + NAME + "_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]

# - #
print "# NAME: " + sweep_name
print "# NUM RUNS: " + str(SWEEP_RUNS)
print "# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS)
print "# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS)
print

# Print training paths as variables so they can be easily changed without
# having to change this script.
print "# Adjust these to your own setup."
print "TRAINING_DATA_PATH=" + FLAGS.training_data_path
print "EVAL_DATA_PATH=" + FLAGS.eval_data_path
print "EMBEDDING_DATA_PATH=" + FLAGS.embedding_data_path
print "LOG_PATH=" + FLAGS.log_path
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

    flags += " --experiment_name " + name
    if NYU_NON_PBS:
        print "cd spinn/python; python2.7 -m spinn.models.supervised_classifier " + flags
    else:
        print "SPINNMODEL=\"spinn.models.supervised_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit.sh ../scripts/train_spinn.sbatch 1"
    print
