# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "05_20"
SWEEP_RUNS = 16

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "spinn/data/listops/train_d20a.tsv", "")
gflags.DEFINE_string("eval_data_path", "spinn/data/listops/test_d20a.tsv", "")
gflags.DEFINE_string("log_path", "/home/sb6065/logs/spinn", "")
gflags.DEFINE_string("metrics_path", "/home/sb6065/logs/spinn-runs", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "data_type":     "listops",
    "model_type":      "RLSPINN",
    "training_data_path":    FLAGS.training_data_path,
    "eval_data_path":    FLAGS.eval_data_path,
    "log_path": FLAGS.log_path,
    "metrics_path": FLAGS.metrics_path,
    "ckpt_path":  FLAGS.log_path,
    "word_embedding_dim":   "32",
    "model_dim":   "32",
    "seq_length":   "100",
    "eval_seq_length":  "3000",
    "eval_interval_steps": "100",
    "statistics_interval_steps": "100",
    "use_internal_parser": "",
    "batch_size":  "64",
    "nouse_tracking_in_composition": "",
    "mlp_dim": "16",
    "num_mlp_layers": "2",
    "transition_weight": "1",
    "embedding_keep_rate": "1.0",
    "semantic_classifier_keep_rate": "1.0",
    "rl_baseline": "greedy",
    "rl_reward": "xent",
    "rl_wake_sleep": "",
    "rl_valid": "",
    "num_samples": "5",
    "nolateral_tracking": "",
    "encode": "pass",
    "norl_catalan": "",
    "norl_wake_sleep": "",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "rl_weight":  ("rlwt", EXP, 0.01, 1),
    "learning_rate":      ("lr", EXP, 0.0006, 0.06),  # RNN likes higher, but below 009.
    "l2_lambda":          ("l2", EXP, 8e-7, 1e-4),
    "learning_rate_decay_per_10k_steps": ("dec", EXP, 0.3, 1.0),
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
        elif t==SS_BASE:
            lmn = np.log(mn)
            lmx = np.log(mx)
            sample = 1 - np.exp(lmn + (lmx - lmn) * r)
        else:
            sample = mn + (mx - mn) * r

        if isinstance(mn, int):
            sample = int(round(sample, 0))
            val_disp = str(sample)
        else: 
            val_disp = "%.2g" % sample

        params[param] = sample
        name += "-" + config[0] + val_disp

    flags = ""
    for param in params:
        value = params[param]
        val_str = ""
        flags += " --" + param + " " + str(value)

    flags += " --experiment_name " + name
    if NYU_NON_PBS:
        print "cd spinn/python; python2.7 -m spinn.models.reinforce " + flags
    else:
        print "SPINNMODEL=\"spinn.models.reinforce\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_cpu_only.sh"
    print
