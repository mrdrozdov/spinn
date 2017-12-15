"""
Converts a model report (report_path) to a Kaggle-style CSV.

Usage:
$ python scripts/report_to_submission_csv.py \
    --report_path ./logs/example-nli.report > submission.csv
"""

import gflags
import sys
import codecs
import json

LABEL_MAP = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

FLAGS = gflags.FLAGS

def run():
    print("pairID,gold_label")
    report = {}
    with codecs.open(FLAGS.report_path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            label = LABEL_MAP[loaded_example['prediction']]
            pair_id = loaded_example['example_id']
            print(pair_id + "," + label)


if __name__ == '__main__':
    gflags.DEFINE_string("report_path", "./checkpoints/example-nli.report", "")

    FLAGS(sys.argv)

    run()
