#!/usr/bin/env python

import os
import json
import codecs

SENTENCE_PAIR_DATA = True

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions

def load_data(path, lowercase=False):
    print "Loading", path
    examples = []
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            example["example_id"] = loaded_example.get('pairID', 'NoID')
            if loaded_example["sentence1_binary_parse"] and loaded_example["sentence2_binary_parse"]:
                (example["premise_tokens"], example["premise_transitions"]) = convert_binary_bracketing(loaded_example["sentence1_binary_parse"], lowercase=lowercase)
                (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert_binary_bracketing(loaded_example["sentence2_binary_parse"], lowercase=lowercase)
                examples.append(example)
            else:
                print("No parse found: {}".format(line.strip()))
                
    return examples, None


if __name__ == "__main__":
    # Demo:
    paths = [
        os.path.expanduser('~/data/multinli_0.1/multinli_0.1_train.jsonl'),
        os.path.expanduser('~/data/multinli_0.1/multinli_0.1_dev_mismatched.jsonl'),
        os.path.expanduser('~/data/multinli_0.1/multinli_0.1_dev_matched.jsonl')
    ]
    for path in paths:
        examples, _ = load_data(path)
        print examples[0]
        print
