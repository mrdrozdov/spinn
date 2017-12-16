# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "07_22_acl"
SWEEP_RUNS = 6

LIN = "LIN"
EXP = "EXP"
BOOL = "BOOL"
CHOICE = "CHOICE"
SS_BASE = "SS_BASE"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "python/spinn/data/listops/train_d20a.tsv", "")
gflags.DEFINE_string("eval_data_path", "python/spinn/data/listops/test_d20a.tsv", "")
gflags.DEFINE_string("log_path", "~/logs/spinn", "")
gflags.DEFINE_string("metrics_path", "~/logs/spinn", "")

FLAGS(sys.argv)

# Instructions: Configure the variables in this block, then run
# the following on a machine with qsub access:
# python make_sweep.py > my_sweep.sh
# bash my_sweep.sh

# - #

# Non-tunable flags that must be passed in.

FIXED_PARAMETERS = {
    "batch_size":  "64",
    "ckpt_path":  FLAGS.log_path,
    "data_type":     "listops",
    "embedding_keep_rate": "1.0",
    "encode": "pass",
    "eval_data_path":    FLAGS.eval_data_path,
    "eval_seq_length":  "3000",
    "gpu":  "0",
    "log_path": FLAGS.log_path,
    "mlp_dim": "16",
    "model_dim":   "32",
    "model_type":      "RLSPINN",
    "nolateral_tracking": "",
    "nouse_tracking_in_composition": "",
    "num_samples": "1",
    "rl_baseline": "value",
    "rl_catalan": "",
    "rl_catalan_backprop": "",
    "rl_reward": "standard",
    "rl_wake_sleep": "",
    "semantic_classifier_keep_rate": "1.0",
    "seq_length":   "100",
    "training_data_path":    FLAGS.training_data_path,
    "transition_weight": "1",
    "use_internal_parser": "",
    "word_embedding_dim":   "32",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "rl_weight":  ("rlwt", EXP, 40., 60.),
    "learning_rate":      ("lr", EXP, 0.004, 0.01),
    "l2_lambda":          ("l2", EXP, 2e-6, 8e-6),
    "learning_rate_decay_when_no_progress": ("dec", EXP, 0.7, 0.9),
    "rl_epsilon": ("eps", LIN, 0.8, 1.),
    "rl_epsilon_decay": ("epsd", LIN, 8000, 10000),
    "rl_confidence_penalty": ("rlconf", EXP, 1.5, 3.),
    "rl_confidence_interval": ("rlconfint", EXP, 100, 700),
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
    cuda = run_id % 2
    print("export CUDA_VISIBLE_DEVICES={}; export PYTHONPATH=./python; cd ~/Developer/spinn; ".format(cuda) + \
        "nohup python -m spinn.models.rl_classifier {} &> nohup_{}.out &".format(flags, name))
    # if NYU_NON_PBS:
    #     print "cd spinn/python; python3 -m spinn.models.rl_classifier " + flags
    # else:
    #     print "SPINNMODEL=\"spinn.models.rl_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_cpu_only.sh"
    print()
