"""
Script that takes a proto log file and spits out a CSV.

=== examples ===
<1> create new data structure:
python spinnLogtoCSV.py spinn-1.log 

<2> create new data structure and save CSV:
python spinnLogtoCSV.py snli-spinn-1.log --save=spinn-1.csv

<3> create new data structure and create a pickle file for pandas dataframe:
python spinnLogtoCSV.py snli-spinn-1.log --dump=spinn-1.p

=== The log file should follow the format ===
entries {
  step: 3690
  class_accuracy: 0.98984375
  transition_accuracy: 0.977040816327
  total_cost: 0.121947198175
  cross_entropy_cost: 0.00855660345405
  transition_cost: 0.0595673844218
  l2_cost: 0.053823210299
  time_per_token_seconds: 0.00471487410094
  learning_rate: 0.0051496206338
  invalid: 0.08671875
}
entries {
  step: 3700
  class_accuracy: 0.99296875
  transition_accuracy: 0.986030629139
  total_cost: 0.1134525612
  cross_entropy_cost: 0.0120047032833
  transition_cost: 0.0472235158086
  l2_cost: 0.0542243421078
  time_per_token_seconds: 0.00537255612064
  learning_rate: 0.00515571678468
  invalid: 0.025
  evaluation {
    eval_class_accuracy: 0.952223557692
    eval_transition_accuracy: 0.861949731036
    filename: "spinn/data/listops/test_d20.tsv"
    time_per_token_seconds: 0.00202515267969
    invalid: 0.106370192308
  }
  rl_sampling {
    t_idx: 0
    crossing: 0
    gold_lb: "22000100101001010110110110101"
    pred_tr: "22000100101001010110110110101"
    pred_ev: "22000100101001010110110110101"
    strg_tr: "00XX9XX9XXXXXXXXXXXXXXXXXXXXX"
    strg_ev: "00XX9XX9XXXXXXXXXXXXXXXXXXXXX"
  }
}

"""

import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("log_path_spinn", type=str, help="Path to log file.")
parser.add_argument("--save", type=str, help="Path and name for csv.")
parser.add_argument("--dump", type=str, help="Path and name for pickle dump.")

args = parser.parse_args()


tr_columns  = ["step", "class_accuracy", "transition_accuracy", "total_cost", "cross_entropy_cost", "transition_cost", "l2_cost"]
ev_columns = ["eval_class_accuracy", "eval_transition_accuracy"]
sample_columns = ["gold_lb", "pred_tr", "pred_ev", "strg_tr", "strg_ev"]
columns = tr_columns + ev_columns + sample_columns


"""
Extract data 
"""

d = {}
for column in columns:
  d["{0}".format(column)]=[]

with open(args.log_path_spinn) as f:
  s = 0
  for line in f:
    start = re.search("entries", line)
    check = re.search("step", line)
    
    # Skip header
    if start:
      s += 1

    if s > 0:
      for val in d:
        cur = re.search(str(val), line)
        ev = re.search("eval", line)
        if cur and ev:
          if val in ev_columns:
            d[val] += re.findall(':(.*)', line)
          else:
            pass
        if cur and not ev:
          d[val] += re.findall(':(.*)', line)

      # Make evaluation columns the same length
      if check and (len(d["step"]) != len(d[ev_columns[0]])):
        for val in ev_columns:
          d[val] += "-"

      # Make sample columns the same length
      if check and (len(d["step"]) != len(d[sample_columns[0]])):
        for val in sample_columns:
          d[val] += "-"


"""
Convert to dataframe and save as csv
"""

fin = pd.DataFrame(d) 

strings = sample_columns # + other columns you may have with strings

for column in strings:
  for item in range(len(fin[column])):
    fin[column][item] = fin[column][item].replace('"', '')

if args.save:
  fin.to_csv(args.save, sep=",")

if args.dump:
  pickle.dump(fin, open(args.dump, 'wb'))
