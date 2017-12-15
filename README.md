# Stack-augmented Parser-Interpreter Neural Network

This repository was used for the paper [A Fast Unified Model for Sentence Parsing and Understanding][1] and [original codebase][9], and is under active development for further projects. For a more informal introduction to the ideas behind the model, see this [Stanford NLP blog post][8].

### Installation

Requirements:

- Python 3.6
- PyTorch 0.3
- Additional dependencies listed in python/requirements.txt

Install PyTorch based on instructions online: http://pytorch.org

Install the other Python dependencies using the command below.

    python3 -m pip install -r python/requirements.txt

### Running the code

The main executable for the SNLI experiments in the paper is [supervised_classifier.py](https://github.com/mrdrozdov/spinn/blob/master/python/spinn/models/supervised_classifier.py), whose flags specify the hyperparameters of the model. You can specify gpu usage by setting `--gpu` flag greater than or equal to 0. Uses the CPU by default.

Here's a sample command that runs a fast, low-dimensional CPU training run, training and testing only on the dev set. It assumes that you have a copy of [SNLI](http://nlp.stanford.edu/projects/snli/) available locally.

        PYTHONPATH=spinn/python \
            python3 -m spinn.models.supervised_classifier --data_type nli \
        --training_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --eval_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --embedding_data_path python/spinn/tests/test_embedding_matrix.5d.txt \
        --word_embedding_dim 5 --model_dim 10 --model_type CBOW

For full runs, you'll also need a copy of the 840B word 300D [GloVe word vectors](http://nlp.stanford.edu/projects/glove/).

## Semi-Supervised Parsing

You can train SPINN using only sentence-level labels. In this case, the integrated parser will randomly sample labels during training time, and will be optimized with the REINFORCE algorithm. The command to run this model looks slightly different:

    python3 -m spinn.models.rl_classifier --data_type listops \
        --training_data_path spinn/python/spinn/data/listops/train_d20a.tsv \
        --eval_data_path spinn/python/spinn/data/listops/test_d20a.tsv  \
        --word_embedding_dim 32 --model_dim 32 --mlp_dim 16 --model_type RLSPINN \
        --rl_baseline value --rl_reward standard --rl_weight 42.0

Note: This model does not yet work well on natural language data, although it does on the included synthetic dataset called `listops`. Please look at the [sweep file][10] for an idea of which hyperparameters to use.

## Log Analysis

This project contains a handful of tools for easier analysis of your model's performance.

For one, after a periodic number of batches, some useful statistics are printed to a file specified by `--log_path`. This is convenient for visual inspection, and the script [parse_logs.py](https://github.com/nyu-mll/spinn/blob/master/scripts/parse_logs.py) is an example of how to easily parse this log file.

## Contributing

If you're interested in proposing a change or fix to SPINN, please submit a Pull Request. In addition, ensure that existing tests pass, and add new tests as you see appropriate. To run tests, simply run this command from the root directory:

    nosetests python/spinn/tests

### Adding Logging Fields

SPINN outputs metrics and statistics into a text [protocol buffer](https://developers.google.com/protocol-buffers/) format. When adding new fields to the proto file, the generated proto code needs to be updated.

    bash python/build.sh

## License

Copyright 2017, New York University

Available for open source use/redistribution. Terms TBD soon. Contact us with questions.
