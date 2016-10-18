"""Dataset handling and related yuck."""

import random
import itertools

import numpy as np
import theano
import time
import sys


# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "*PADDING*"

# Temporary hack: Map UNK to "_" when loading pretrained embedding matrices:
# it's a common token that is pretrained, but shouldn't look like any content words.
UNK_TOKEN = "_"

CORE_VOCABULARY = {PADDING_TOKEN: 0,
                   UNK_TOKEN: 1}

# Allowed number of transition types : currently PUSH : 0 and MERGE : 1
NUM_TRANSITION_TYPES = 2


def TrimDataset(dataset, seq_length, eval_mode=False, sentence_pair_data=False):
    """Avoid using excessively long training examples."""
    if eval_mode:
        return dataset
    else:
        if sentence_pair_data:
            new_dataset = [example for example in dataset if
                len(example["premise_transitions"]) <= seq_length and
                len(example["hypothesis_transitions"]) <= seq_length]
        else:
            new_dataset = [example for example in dataset if len(
                example["transitions"]) <= seq_length]
        return new_dataset


def TokensToIDs(vocabulary, dataset, sentence_pair_data=False):
    """Replace strings in original boolean dataset with token IDs."""
    if sentence_pair_data:
        keys = ["premise_tokens", "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for key in keys:
        if UNK_TOKEN in vocabulary:
            unk_id = vocabulary[UNK_TOKEN]
            for example in dataset:
                example[key] = [vocabulary.get(token, unk_id)
                                     for token in example[key]]
        else:
            for example in dataset:
                example[key] = [vocabulary[token]
                                for token in example[key]]
    return dataset


def MakeTrainingIterator(sources, batch_size):
    # Make an iterator that exposes a dataset as random minibatches.
    sources = np.array(sources).T
    def data_iter():
        dataset_size = len(sources[0])
        start = -1 * batch_size
        order = range(dataset_size)
        random.shuffle(order)

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            batch = sources[batch_indices].T.tolist()
            batch = [np.array(s) for s in batch]
            yield batch
    return data_iter()


def MakeEvalIterator(sources, batch_size, limit=-1):
    # Make a list of minibatches from a dataset to use as an iterator.
    # TODO(SB): Pad out the last few examples in the eval set if they don't
    # form a batch.

    print "WARNING: May be discarding eval examples."

    dataset_size = limit if limit > 0 else len(sources[0])
    data_iter = []
    sources = np.array(sources).T

    start = -batch_size

    while True:
        start += batch_size
        if start >= dataset_size:
            break

        # batch_indices = range(start, min(start + batch_size, dataset_size))
        candidate_batch = sources[start:start+batch_size].T.tolist()
        candidate_batch = [np.array(s) for s in candidate_batch]

        if len(candidate_batch[0]) == batch_size:
            data_iter.append(candidate_batch)
        else:
            print "Skipping " + str(len(candidate_batch[0])) + " examples."

    return data_iter


def NaiveCropAndPad(example, length, symbol=-1):
    example_len = len(example)
    diff_len = length - len(example)
    if diff_len <= 0:
        return example[:length]
    else:
        return [symbol] * diff_len + example


def PreprocessDataset(dataset, vocabulary, seq_length, data_manager, eval_mode=False, logger=None,
                      sentence_pair_data=False, for_rnn=False):
    dataset = TrimDataset(dataset, seq_length, eval_mode=eval_mode, sentence_pair_data=sentence_pair_data)
    dataset = TokensToIDs(vocabulary, dataset, sentence_pair_data=sentence_pair_data)

    X_prem = [NaiveCropAndPad(example["premise_tokens"], seq_length, symbol=0) for example in dataset]
    X_hyp  = [NaiveCropAndPad(example["hypothesis_tokens"], seq_length, symbol=0) for example in dataset]
    nt_prem = [len(example["premise_transitions"]) for example in dataset]
    nt_hyp  = [len(example["hypothesis_transitions"]) for example in dataset]
    t_prem  = [NaiveCropAndPad(example["premise_transitions"], seq_length) for example in dataset]
    t_hyp   = [NaiveCropAndPad(example["hypothesis_transitions"], seq_length) for example in dataset]
    y = [data_manager.LABEL_MAP[example["label"]] for example in dataset]

    # TODO: These types should probably set more elegantly.
    X_prem = [np.array(t, dtype=np.int32) for t in X_prem]
    X_hyp = [np.array(t, dtype=np.int32) for t in X_hyp]
    t_prem = [np.array(t, dtype=np.int32) for t in t_prem]
    t_hyp = [np.array(t, dtype=np.int32) for t in t_hyp]
    y = [np.array(t, dtype=np.int32) for t in y]

    return X_prem, X_hyp, t_prem, t_hyp, y, nt_prem, nt_hyp


def BuildVocabulary(raw_training_data, raw_eval_sets, embedding_path, logger=None, sentence_pair_data=False):
    # Find the set of words that occur in the data.
    logger.Log("Constructing vocabulary...")
    types_in_data = set()
    for dataset in [raw_training_data] + [eval_dataset[1] for eval_dataset in raw_eval_sets]:
        if sentence_pair_data:
            types_in_data.update(itertools.chain.from_iterable([example["premise_tokens"]
                                                                for example in dataset]))
            types_in_data.update(itertools.chain.from_iterable([example["hypothesis_tokens"]
                                                                for example in dataset]))
        else:
            types_in_data.update(itertools.chain.from_iterable([example["tokens"]
                                                                for example in dataset]))
    logger.Log("Found " + str(len(types_in_data)) + " word types.")

    if embedding_path == None:
        logger.Log(
            "Warning: Open-vocabulary models require pretrained vectors. Running with empty vocabulary.")
        vocabulary = CORE_VOCABULARY
    else:
        # Build a vocabulary of words in the data for which we have an
        # embedding.
        vocabulary = BuildVocabularyForASCIIEmbeddingFile(
            embedding_path, types_in_data, CORE_VOCABULARY)

    return vocabulary


def BuildVocabularyForASCIIEmbeddingFile(path, types_in_data, core_vocabulary):
    """Quickly iterates through a GloVe-formatted ASCII vector file to
    extract a working vocabulary of words that occur both in the data and
    in the vector file."""

    # TODO(SB): Report on *which* words are skipped. See if any are common.

    vocabulary = {}
    vocabulary.update(core_vocabulary)
    next_index = len(vocabulary)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ", 1)
            word = spl[0]
            if word in types_in_data:
                vocabulary[word] = next_index
                next_index += 1
    return vocabulary


def LoadEmbeddingsFromASCII(vocabulary, embedding_dim, path):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a GloVe - format ASCII vector file.

    For now, values not found in the file will be set to zero."""
    emb = np.zeros(
        (len(vocabulary), embedding_dim), dtype=theano.config.floatX)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ")
            word = spl[0]
            if word in vocabulary:
                emb[vocabulary[word], :] = [float(e) for e in spl[1:]]
    return emb


def TransitionsToParse(transitions, words):
    if transitions is not None:
        stack = ["(P *ZEROS*)"] * (len(transitions) + 1)
        buffer_ptr = 0
        for transition in transitions:
            if transition == 0:
                stack.append("(P " + words[buffer_ptr] +")")
                buffer_ptr += 1
            elif transition == 1:
                r = stack.pop()
                l = stack.pop()
                stack.append("(M " + l + " " + r + ")")
        return stack.pop()
    else:
        return " ".join(words)


class SimpleProgressBar(object):
    """ Simple Progress Bar and Timing Snippet
    """

    def __init__(self, msg=">", bar_length=80):
        super(SimpleProgressBar, self).__init__()
        self.begin = time.time()
        self.bar_length = bar_length
        self.msg = msg
        
    def step(self, i, total):
        sys.stdout.write('\r')
        pct = (i / float(total)) * 100
        ii = i * self.bar_length / total
        fmt = "%s [%-{}s] %d%% %ds / %ds".format(self.bar_length)
        total_time = time.time()-self.begin
        expected = total_time / ((i+1e-03) / float(total))
        sys.stdout.write(fmt % (self.msg, '='*ii, pct, total_time, expected))
        sys.stdout.flush()

    def finish(self):
        self.begin = time.time()
        sys.stdout.write('\n')
