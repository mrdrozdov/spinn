# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "06_21"
SWEEP_RUNS = 8

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
    "model_type":      "Pyramid",
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "embedding_data_path": FLAGS.embedding_data_path,
    "log_path": FLAGS.log_path,
    "metrics_path": FLAGS.log_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "300",
    "model_dim":   "300",
    "seq_length":   "80",
    "eval_seq_length":  "810",
    "eval_interval_steps": "1000",
    "statistics_interval_steps": "100",
    "semantic_classifier_keep_rate": "1.0",
    "embedding_keep_rate": "1.0",
    "batch_size":  "128",
    "encode": "gru",
    "encode_bidirectional": "",
    "num_mlp_layers": "2",
    "use_internal_parser": "",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "mlp_dim":      ("mld", EXP, 96, 256),  # RNN likes higher, but below 009.
    "semantic_classifier_keep_rate": ("skr", LIN, 0.8, 1.0),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": ("ekr", LIN, 0.8, 1.0),
    "seq_length":      ("seq", LIN, 40, 120),  # RNN likes higher, but below 009.
    "learning_rate":      ("lr", EXP, 0.00005, 0.002),  # RNN likes higher, but below 009.
    "l2_lambda":          ("l2", EXP, 1e-7, 1e-3),
    "learning_rate_decay_per_10k_steps": ("dc", LIN, 0.4, 1.0),
    "pyramid_trainable_temperature": ("tt", BOOL, None, None),
    "pyramid_temperature_decay_per_10k_steps": ("tdc", EXP, 0.33, 1.0),
    "pyramid_selection_dim": ("sd", EXP, 2, 64),
    "pyramid_temperature_cycle_length": ("cl", CHOICE, ['0', '0', '30', '300'], None),
    "pyramid_gumbel": ("pg", CHOICE, ['plain', 'st'], None),   
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
        print "SPINNMODEL=\"spinn.models.supervised_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit.sh"
    print
