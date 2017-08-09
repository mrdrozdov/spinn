""" Script to combine different jsonl nli files.

$ python scripts/combine_nli.py
Reading: /Users/Andrew/data/multinli_0.9/multinli_0.9_train.jsonl
Read 392702 lines. Kept 365990.
Reading: /Users/Andrew/data/snli_1.0/snli_1.0_train.jsonl
Read 550152 lines. Kept 549263.
"""

import gflags
import sys
import os
import json

FLAGS = gflags.FLAGS

gflags.DEFINE_string("inputs", "~/data/multinli_0.9/multinli_0.9_train.jsonl,~/data/snli_1.0/snli_1.0_train.jsonl", "")
gflags.DEFINE_string("output", "~/data/nli_train_combined.jsonl", "")
gflags.DEFINE_integer("seq_length", 80, "")

FLAGS(sys.argv)

db = dict()

def sentence_okay(sentence):
    # This is probably overly lenient depending on how your model does tokenization.
    num_tokens = len(sentence.split(' '))
    num_transitions = num_tokens * 2 - 1
    return num_transitions < FLAGS.seq_length

def is_okay(line):
    obj = json.loads(line)
    sentence1, sentence2 = obj['sentence1'], obj['sentence2']
    return sentence_okay(sentence1) and sentence_okay(sentence2)

for filename in FLAGS.inputs.split(','):
    filename = os.path.expanduser(filename)
    print("Reading: {}".format(filename))
    db[filename] = []
    with open(filename) as f:
        for ii, line in enumerate(f):
            if is_okay(line):
                db[filename].append(line)

    print("Read {} lines. Kept {}.".format(ii + 1, len(db[filename])))

output = os.path.expanduser(FLAGS.output)
with open(output, 'w') as f_out:
    for filename in db.keys():
        with open(filename) as f_in:
            for line in f_in:
                f_out.write(line)
