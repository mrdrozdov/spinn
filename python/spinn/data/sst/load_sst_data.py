#!/usr/bin/env python

# Loads a file where each line contains a label, followed by a tab, followed
# by a sequence of words with a binary parse indicated by space-separated parentheses.
#
# Example:
# sentence_label	( ( word word ) ( ( word word ) word ) )

import collections
import numpy as np
import sys

from spinn import util

SENTENCE_PAIR_DATA = False

LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4
}


def convert_unary_binary_bracketed_data(filename):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    examples = []
    with open(filename, 'r') as f:
        for line in f:
            example = {}
            line = line.strip()
            if len(line) == 0:
                continue
            example["label"] = line[1]
            example["sentence"] = line
            example["tokens"] = []
            example["transitions"] = []

            words = example["sentence"].split(' ')
            for index, word in enumerate(words):
                if word[0] != "(":
                    if word == ")":  
                        # Ignore unary merges
                        if words[index - 1] == ")":
                            example["transitions"].append(1)
                    else:
                        # Downcase all words to match GloVe.
                        example["tokens"].append(word.lower())
                        example["transitions"].append(0)
            examples.append(example)
    return examples


def load_data(path, vocabulary=None, seq_length=None, batch_size=32, eval_mode=False, logger=None):
    dataset = convert_unary_binary_bracketed_data(path)
    return dataset, None


def create_binary_classes(input_path, output_path):
    with open(input_path) as f, open(output_path, 'w') as f_out:
        for line in f:
            if int(line[1]) < 2:
                line = line[0] + '0' + line[2:]
                f_out.write(line)
            elif int(line[1]) > 2:
                line = line[0] + '1' + line[2:]
                f_out.write(line)
            else:
                # skip neutral sentences
                pass


if __name__ == "__main__":
    # Demo:
    if len(sys.argv) > 1 and sys.argv[1] == 'binary':
        create_binary_classes('sst/dev.txt', 'sst/dev-binary.txt')
        create_binary_classes('sst/test.txt', 'sst/test-binary.txt')
        create_binary_classes('sst/train.txt', 'sst/train-binary.txt')
    else:
        examples = convert_unary_binary_bracketed_data('sst/dev.txt')
        print examples[0]
