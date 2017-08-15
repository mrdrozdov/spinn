"""
If you don't know what it is, it won't help you.

Usage:
$ python scripts/parse_comparison.py \
    --read_path ptb.txt \
    --write_path ptb.jsonl \
"""

import gflags
import sys
import codecs
import json

FLAGS = gflags.FLAGS

def run():
    with open(FLAGS.write_path, 'w') as outfile:
        with codecs.open(FLAGS.read_path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    line = line.encode('UTF-8')
                except UnicodeError as e:
                    print "ENCODING ERROR:", line, e
                    line = "{}"
                item = {
                    'genre': 'ptb_wsj',
                    'gold_label': 'neutral',
                    'pairID': 'ptb' + str(idx),
                    'sentence1': '-',
                    'sentence2': '-',
                    'sentence1_binary_parse': line.strip(),
                    'sentence2_binary_parse': '-',
                }
                json.dump(item, outfile, sort_keys=True)
                outfile.write('\n')

if __name__ == '__main__':
    gflags.DEFINE_string("read_path", "/Users/Bowman/Drive/Parser/stanford-corenlp-full-2015-04-20/wsj.txt", "")
    gflags.DEFINE_string("write_path", "ptb.jsonl", "")

    FLAGS(sys.argv)

    run()
