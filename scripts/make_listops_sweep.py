# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "listops_07_24"
SWEEP_RUNS = 8

LIN = "LIN"
EXP = "EXP"
BOOL = "BOOL"
CHOICE = "CHOICE"
SS_BASE = "SS_BASE"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "spinn/data/listops/train_d20s.tsv", "")
gflags.DEFINE_string("eval_data_path", "spinn/data/listops/test_d20s.tsv", "")
gflags.DEFINE_string("log_path", "/scratch/sb6065/logs", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "listops",
    "model_type":      "ChoiPyramid",
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "log_path": FLAGS.log_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "128",
    "model_dim":   "128",
    "eval_seq_length":  "500",
    "allow_eval_cropping": "",
    "eval_interval_steps": "1000",
    "statistics_interval_steps": "100",
    "batch_size":  "64",
    "encode": "pass",
    "mlp_dim": "16",
    "num_mlp_layers": "2",
    "semantic_classifier_keep_rate": "1.0",
    "embedding_keep_rate": "1.0",
    "sample_interval_steps": "1000",
    "pyramid_test_time_temperature_multiplier": "0.0",
    "nocomposition_ln": "",
    "learning_rate": "0.001",
    "seq_length": "100",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "l2_lambda":          ("l2", EXP, 8e-7, 1e-3),
    "learning_rate_decay_when_no_progress": ("dc", LIN, 0.3, 1.0),
    "pyramid_trainable_temperature": ("tt", BOOL, None, None),
    "pyramid_temperature_decay_per_10k_steps": ("tdc", EXP, 0.2, 1.0),
    "pyramid_temperature_cycle_length": ("cl", CHOICE, ['0', '0', '300', '3000'], None),
    "learning_rate": ("lr", EXP, 0.0001, 0.01),
}

sweep_name = "sweep_" + NAME + "_" + \
    FIXED_PARAMETERS["data_type"] + "_" + FIXED_PARAMETERS["model_type"]

# - #
print("# NAME: " + sweep_name)
print("# NUM RUNS: " + str(SWEEP_RUNS))
print("# SWEEP PARAMETERS: " + str(SWEEP_PARAMETERS))
print("# FIXED_PARAMETERS: " + str(FIXED_PARAMETERS))
print()

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
        print("cd spinn/python; python2.7 -m spinn.models.supervised_classifier " + flags)
    else:
        print("SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_cpu_only.sh")
    print()
