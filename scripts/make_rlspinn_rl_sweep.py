# Create a script to run a random hyperparameter search.

import copy
import getpass
import os
import random
import numpy as np
import gflags
import sys

NYU_NON_PBS = False
NAME = "07_30_tacl"
SWEEP_RUNS = 4

LIN = "LIN"
EXP = "EXP"
SS_BASE = "SS_BASE"
BOOL = "BOOL"
CHOICE = "CHOICE"

FLAGS = gflags.FLAGS

gflags.DEFINE_string("training_data_path", "~/data/nli_train_combined.jsonl", "")
gflags.DEFINE_string("eval_data_path", "~/data/multinli_0.9/multinli_0.9_dev_matched.jsonl", "")
gflags.DEFINE_string("embedding_data_path", "~/data/glove/glove.840B.300d.txt", "")
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
    "batch_size":  "32",
    "ckpt_path":  FLAGS.log_path,
    "data_type":     "nli",
    "embedding_data_path": FLAGS.embedding_data_path,
    "encode": "gru",
    "encode_bidirectional": "",
    "eval_data_path":    FLAGS.eval_data_path,
    "eval_interval_steps": "1000",
    "eval_seq_length":  "810",
    "log_path": FLAGS.log_path,
    "metrics_path": FLAGS.log_path,
    "mlp_dim": "1024",
    "model_dim":   "300",
    "model_type":      "RLSPINN",
    "norl_wake_sleep": "",
    "num_mlp_layers": "1",
    "num_samples": "1",
    "rl_baseline": "value",
    "rl_reward": "standard",
    "seq_length":   "80",
    "statistics_interval_steps": "100",
    "sample_interval_steps": "1000",
    "training_data_path":    FLAGS.training_data_path,
    "transition_weight": "1.0",
    "use_internal_parser": "",
    "word_embedding_dim":   "300",
    "learning_rate_decay_per_10k_steps": "0.906176807691",
}

# Tunable parameters.
SWEEP_PARAMETERS = {
    "learning_rate":      ("lr", EXP, 0.00003, 0.01),  # RNN likes higher, but below 009.
    "rl_weight":  ("rlwt", EXP, 1.0, 5.0),
    "l2_lambda":          ("l2", EXP, 1e-7, 1e-5),
    "tracking_lstm_hidden_dim": ("tdim", EXP, 8, 64),
    "semantic_classifier_keep_rate": ("skr", LIN, 0.8, 1.0),  # NB: Keep rates may depend considerably on dims.
    "embedding_keep_rate": ("ekr", LIN, 0.8, 1.0),
    "rl_epsilon": ("eps", LIN, 0.1, 1.0),
    "rl_epsilon_decay": ("epsd", EXP, 1000, 1000000),
    "rl_confidence_penalty": ("rlcp", EXP, 0.00001, 0.01),
    "rl_confidence_interval": ("rlci", EXP, 10, 100),
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
    cuda = run_id % 2
    print "export CUDA_VISIBLE_DEVICES={}; export PYTHONPATH=./python; cd ~/Developer/spinn; ".format(cuda) + \
        "nohup python -m spinn.models.rl_classifier {} &> nohup_{}.out &".format(flags, name)
    # if NYU_NON_PBS:
    #     print "cd spinn/python; python2.7 -m spinn.models.rl_classifier " + flags
    # else:
    #     print "SPINNMODEL=\"spinn.models.rl_classifier\" SPINN_FLAGS=\"" + flags + "\" bash ../scripts/sbatch_submit_cpu_only.sh"
    print
