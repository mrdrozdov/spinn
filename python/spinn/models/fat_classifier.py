"""From the project root directory (containing data files), this can be run with:

Boolean logic evaluation:
python -m spinn.models.fat_classifier --training_data_path ../bl-data/pbl_train.tsv \
       --eval_data_path ../bl-data/pbl_dev.tsv

SST sentiment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type sst --training_data_path sst-data/train.txt \
       --eval_data_path sst-data/dev.txt --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

SNLI entailment (Demo only, model needs a full GloVe embeddings file to do well):
python -m spinn.models.fat_classifier --data_type snli --training_data_path snli_1.0/snli_1.0_dev.jsonl \
       --eval_data_path snli_1.0/snli_1.0_dev.jsonl --embedding_data_path spinn/tests/test_embedding_matrix.5d.txt \
       --model_dim 10 --word_embedding_dim 5

Note: If you get an error starting with "TypeError: ('Wrong number of dimensions..." during development,
    there may already be a saved checkpoint in ckpt_path that matches the name of the model you're developing.
    Move or delete it as appropriate.
"""

import os
import pprint
import sys
import time
from collections import deque

import gflags
import numpy as np

from spinn import afs_safe_logger
from spinn import util
from spinn.data.arithmetic import load_simple_data
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import the_gpu, l2_cost, flatten
from spinn.util.misc import Accumulator, time_per_token

import spinn.rl_spinn
import spinn.fat_stack
import spinn.plain_rnn
import spinn.cbow

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


FLAGS = gflags.FLAGS


def build_rewards(logits, y, xent_reward=False):
    if xent_reward:
        return np.mean(logits.data[np.arange(y.shape[0]), y])
    else:
        return metrics.accuracy_score(logits.data.argmax(axis=1), y)


def truncate(X_batch, transitions_batch, num_transitions_batch):
    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = not FLAGS.use_left_padding
    transitions_batch_is_left_padded = FLAGS.use_left_padding
    max_transitions = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    if X_batch_is_left_padded:
        X_batch = X_batch[:, seq_length - max_transitions:]
    else:
        X_batch = X_batch[:, :max_transitions]

    if transitions_batch_is_left_padded:
        transitions_batch = transitions_batch[:, seq_length - max_transitions:]
    else:
        transitions_batch = transitions_batch[:, :max_transitions]

    return X_batch, transitions_batch


def evaluate(classifier_trainer, eval_set, logger, step, vocabulary=None):
    filename, dataset = eval_set

    # Evaluate
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    start = time.time()

    model = classifier_trainer.model

    model.eval()

    transition_preds = []
    transition_targets = []

    for i, (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch) in enumerate(dataset):
        if FLAGS.truncate_eval_batch:
            eval_X_batch, eval_transitions_batch = truncate(
                eval_X_batch, eval_transitions_batch, eval_num_transitions_batch)

        # Run model.
        output = classifier_trainer.forward({
            "sentences": eval_X_batch,
            "transitions": eval_transitions_batch,
            }, eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions)

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()
        pred = logits.data.max(1)[1] # get the index of the max log-probability
        class_acc = pred.eq(target).sum() / float(target.size(0))

        # Optionally calculate transition loss/acc.
        transition_acc = model.transition_acc if hasattr(model, 'transition_acc') else 0.0
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        acc_accum += class_acc
        eval_batches += 1.0
        total_tokens += eval_num_transitions_batch.ravel().sum()

        # Accumulate stats for transition accuracy.
        if transition_loss is not None:
            transition_preds.append([m["t_preds"] for m in model.spinn.memories])
            transition_targets.append([m["t_given"] for m in model.spinn.memories])

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()

    end = time.time()
    total_time = end - start

    # Get time per token.
    time_metric = time_per_token([total_tokens], [total_time])

    # Get average class accuracy.
    avg_class_acc = acc_accum / eval_batches

    # Get average transition accuracy if applicable.
    if len(transition_preds) > 0:
        all_preds = np.array(flatten(transition_preds))
        all_truth = np.array(flatten(transition_targets))
        avg_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])
    else:
        avg_trans_acc = 0.0

    logger.Log("Step: %i\tEval acc: %f\t %f\t%s Time: %5f" %
              (step, avg_class_acc, avg_trans_acc, filename, time_metric))
    return avg_class_acc


def reinforce(optimizer, lr, baseline, mu, reward, transition_loss):
    new_lr = (lr*(reward - baseline))
    baseline = baseline*(1-mu)+mu*reward

    transition_loss.backward()
    transition_loss.unchain_backward()
    optimizer.lr = new_lr
    optimizer.step()

    return new_lr, baseline


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    # Select data format.
    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data
    elif FLAGS.data_type == "arithmetic":
        data_manager = load_simple_data
    else:
        logger.Log("Bad data type.")
        return

    pp = pprint.PrettyPrinter(indent=4)
    logger.Log("Flag values:\n" + pp.pformat(FLAGS.FlagValuesDict()))

    # Load the data.
    raw_training_data, vocabulary = data_manager.load_data(
        FLAGS.training_data_path)

    # Load the eval data.
    raw_eval_sets = []
    if FLAGS.eval_data_path:
        for eval_filename in FLAGS.eval_data_path.split(":"):
            eval_data, _ = data_manager.load_data(eval_filename)
            raw_eval_sets.append((eval_filename, eval_data))

    # Prepare the vocabulary.
    if not vocabulary:
        logger.Log("In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        train_embeddings = False
        vocabulary = util.BuildVocabulary(
            raw_training_data, raw_eval_sets, FLAGS.embedding_data_path, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        logger.Log("In fixed vocabulary mode. Training embeddings.")
        train_embeddings = True

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW",
        use_left_padding=FLAGS.use_left_padding)
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano)

    # Preprocess eval sets.
    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        e_X, e_transitions, e_y, e_num_transitions = util.PreprocessDataset(
            raw_eval_set, vocabulary,
            FLAGS.eval_seq_length if FLAGS.eval_seq_length is not None else FLAGS.seq_length,
            data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW",
            use_left_padding=FLAGS.use_left_padding)
        eval_it = util.MakeEvalIterator((e_X, e_transitions, e_y, e_num_transitions),
            FLAGS.batch_size, FLAGS.eval_data_limit, bucket_eval=FLAGS.bucket_eval,
            shuffle=FLAGS.shuffle_eval, rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))

    # Choose model.
    logger.Log("Building model.")
    if FLAGS.model_type == "CBOW":
        model_module = spinn.cbow
    elif FLAGS.model_type == "RNN":
        model_module = spinn.plain_rnn
    elif FLAGS.model_type == "SPINN":
        model_module = spinn.fat_stack
    elif FLAGS.model_type == "RLSPINN":
        model_module = spinn.rl_spinn
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)

    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(data_manager.LABEL_MAP)

    if data_manager.SENTENCE_PAIR_DATA:
        trainer_cls = model_module.SentencePairTrainer
        model_cls = model_module.SentencePairModel
        use_sentence_pair = True
    else:
        trainer_cls = model_module.SentenceTrainer
        model_cls = model_module.SentenceModel
        num_classes = len(data_manager.LABEL_MAP)
        use_sentence_pair = False

    model = model_cls(FLAGS.model_dim, FLAGS.word_embedding_dim, vocab_size,
         initial_embeddings, num_classes, mlp_dim=FLAGS.mlp_dim,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         tracker_dropout_rate=FLAGS.tracker_dropout_rate,
         use_tracker_dropout=FLAGS.use_tracker_dropout,
         tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
         transition_weight=FLAGS.transition_weight,
         use_tracking_lstm=FLAGS.use_tracking_lstm,
         use_shift_composition=FLAGS.use_shift_composition,
         use_sentence_pair=use_sentence_pair,
         use_skips=FLAGS.use_skips,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
        )

    # Build optimizer.
    if FLAGS.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    elif FLAGS.optimizer_type == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.learning_rate, eps=1e-08)
    else:
        raise NotImplementedError

    # Build trainer.
    classifier_trainer = trainer_cls(model, optimizer)

    # Set checkpoint path.
    if ".ckpt" in FLAGS.ckpt_path:
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + ".ckpt")
    
    # Load checkpoint if available.
    if os.path.isfile(checkpoint_path):
        # TODO: Check that resuming works fine with tf summaries.
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = classifier_trainer.load(checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    # Print model size.
    logger.Log("Architecture: {}".format(model))
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0) for w in model.parameters()])
    logger.Log("Total params: {}".format(total_params))

    # GPU support.
    the_gpu.gpu = FLAGS.gpu
    if FLAGS.gpu >= 0:
        model.cuda()
    else:
        model.cpu()

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)

    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)

    # Do an evaluation-only run.
    if only_forward:
        for index, eval_set in enumerate(eval_iterators):
            acc = evaluate(classifier_trainer, eval_set, logger, step, vocabulary)
    else:
         # Train
        logger.Log("Training.")

        # if FLAGS.use_reinforce:
        #     optimizer_lr = 0.01
        #     baseline = 0
        #     mu = 0.1
        #     transition_optimizer = optimizers.SGD(lr=optimizer_lr)
        #     transition_optimizer.setup(model.spinn.tracker)

        # New Training Loop
        progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
        progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

        for step in range(step, FLAGS.training_steps):
            model.train()

            start = time.time()

            X_batch, transitions_batch, y_batch, num_transitions_batch = training_data_iter.next()

            if FLAGS.truncate_train_batch:
                X_batch, transitions_batch = truncate(
                    X_batch, transitions_batch, num_transitions_batch)

            total_tokens = num_transitions_batch.ravel().sum()

            # Reset cached gradients.
            optimizer.zero_grad()

            # Run model.
            output = classifier_trainer.forward({
                "sentences": X_batch,
                "transitions": transitions_batch,
                }, y_batch,
                use_internal_parser=FLAGS.use_internal_parser,
                validate_transitions=FLAGS.validate_transitions
                )

            # Normalize output.
            logits = F.log_softmax(output)

            # Calculate class accuracy.
            target = torch.from_numpy(y_batch).long()
            pred = logits.data.max(1)[1] # get the index of the max log-probability
            class_acc = pred.eq(target).sum() / float(target.size(0))

            A.add('class_acc', class_acc)

            # Calculate class loss.
            xent_loss = nn.NLLLoss()(logits, Variable(target, volatile=False))

            # Optionally calculate transition loss/accuracy.
            transition_acc = model.transition_acc if hasattr(model, 'transition_acc') else 0.0
            transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None
            rl_loss = model.rl_loss if hasattr(model, 'rl_loss') else None

            # Accumulate stats for transition accuracy.
            if transition_loss is not None:
                preds = [m["t_preds"] for m in model.spinn.memories]
                truth = [m["t_given"] for m in model.spinn.memories]
                A.add('preds', preds)
                A.add('truth', truth)

            # Extract L2 Cost
            l2_loss = l2_cost(model, FLAGS.l2_lambda)

            # Boilerplate for calculating loss values.
            xent_cost_val = xent_loss.data[0]
            transition_cost_val = transition_loss.data[0] if transition_loss is not None else 0.0
            l2_cost_val = l2_loss.data[0]
            rl_cost_val = rl_loss.data[0] if rl_loss is not None else 0.0

            # Accumulate Total Loss Data
            total_cost_val = 0.0
            total_cost_val += xent_loss.data[0]
            if transition_loss is not None and model.optimize_transition_loss:
                total_cost_val += transition_cost_val
            total_cost_val += l2_loss.data[0]
            total_cost_val += rl_cost_val

            # Accumulate Total Loss Variable
            total_loss = 0.0
            total_loss += xent_loss
            total_loss += l2_loss
            if transition_loss is not None and model.optimize_transition_loss:
                total_loss += transition_loss
            if rl_loss is not None:
                total_loss += rl_loss

            # Backward pass.
            total_loss.backward()

            # Hard Gradient Clipping
            clip = FLAGS.clipping_max_value
            for p in model.parameters():
                if p.requires_grad:
                    p.grad.data.clamp_(min=-clip, max=clip)

            # Learning Rate Decay
            if FLAGS.actively_decay_learning_rate:
                optimizer.lr = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

            # Gradient descent step.
            optimizer.step()

            # if FLAGS.use_reinforce:
            #     transition_optimizer.zero_grads()
            #     optimizer_lr, baseline = reinforce(transition_optimizer, optimizer_lr, baseline, mu, rewards, transition_loss)

            end = time.time()

            total_time = end - start

            A.add('total_tokens', total_tokens)
            A.add('total_time', total_time)

            if step % FLAGS.statistics_interval_steps == 0:
                progress_bar.step(i=FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)
                progress_bar.finish()
                avg_class_acc = A.get_avg('class_acc')
                if transition_loss is not None:
                    all_preds = np.array(flatten(A.get('preds')))
                    all_truth = np.array(flatten(A.get('truth')))
                    avg_trans_acc = (all_preds == all_truth).sum() / float(all_truth.shape[0])
                else:
                    avg_trans_acc = 0.0
                time_metric = time_per_token(A.get('total_tokens'), A.get('total_time'))
                stats_args = {
                    "step": step,
                    "class_acc": avg_class_acc,
                    "transition_acc": avg_trans_acc,
                    "total_cost": total_cost_val,
                    "xent_cost": xent_cost_val,
                    "transition_cost": transition_cost_val,
                    "l2_cost": l2_cost_val,
                    "rl_cost": rl_cost_val,
                    "time": time_metric,
                }
                stats_str = "Step: {step}\tAcc: {class_acc:.5f}\t{transition_acc:.5f}\tCost: {total_cost:.5f} {xent_cost:.5f} {transition_cost:.5f} {l2_cost:.5f}"
                if rl_loss is not None:
                    stats_str += " {rl_cost:.5f}"
                stats_str += " Time: {time:.5f}"
                logger.Log(stats_str.format(**stats_args))

            if step > 0 and step % FLAGS.eval_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(classifier_trainer, eval_set, logger, step)
                    # TODO: Should ckpt every X steps.
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > FLAGS.ckpt_step:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        classifier_trainer.save(checkpoint_path + "_best", step, best_dev_error)
                progress_bar.reset()

            elif step % FLAGS.ckpt_interval_steps == 0 and step > FLAGS.ckpt_step:
                logger.Log("Checkpointing.")
                classifier_trainer.save(checkpoint_path, step, best_dev_error)

            progress_bar.step(i=step % FLAGS.statistics_interval_steps, total=FLAGS.statistics_interval_steps)


if __name__ == '__main__':
    # Debug settings.
    gflags.DEFINE_bool("debug", False, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_string("branch_name", "", "")
    gflags.DEFINE_string("sha", "", "")

    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "snli", "arithmetic"],
        "Which data handler and classifier to use.")

    # Where to store checkpoints
    gflags.DEFINE_string("ckpt_path", ".", "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string("log_path", ".", "A directory in which to write logs.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("ckpt_step", 1000, "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_integer("deque_length", None, "Max trailing examples to use for statistics.")
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", None, "")
    gflags.DEFINE_boolean("truncate_eval_batch", True, "Shorten batches to max transition length.")
    gflags.DEFINE_boolean("truncate_train_batch", True, "Shorten batches to max transition length.")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_boolean("bucket_eval", True, "Bucket evaluation data for speed improvement.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer("shuffle_eval_seed", 123, "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN", ["CBOW", "RNN", "SPINN", "RLSPINN"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_float("transition_weight", None, "")
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", 4, "")

    gflags.DEFINE_boolean("xent_reward", False, "Use cross entropy instead of accuracy as RL reward")

    gflags.DEFINE_boolean("use_shift_composition", True, "")
    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")
    gflags.DEFINE_boolean("use_skips", False, "Pad transitions with SKIP actions.")
    gflags.DEFINE_boolean("use_left_padding", True, "Pad transitions only on the LHS.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean("validate_transitions", True, "Constrain predicted transitions to ones"
                                                        "that give a valid parse tree.")
    gflags.DEFINE_boolean("use_tracking_lstm", True,
                          "Whether to use LSTM in the tracking unit")
    gflags.DEFINE_float("embedding_keep_rate", 0.9,
        "Used for dropout on transformed embeddings.")
    gflags.DEFINE_boolean("use_tracker_dropout", False, "")
    gflags.DEFINE_float("tracker_dropout_rate", 0.1, "")
    gflags.DEFINE_boolean("lstm_composition", True, "")

    # MLP settings.
    gflags.DEFINE_integer("mlp_dim", 1024, "")
    gflags.DEFINE_integer("num_mlp_layers", 2, "")
    gflags.DEFINE_boolean("mlp_bn", True, "")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
        "Used for dropout in the semantic task classifier.")

    # Optimization settings.
    gflags.DEFINE_enum("optimizer_type", "Adam", ["Adam", "RMSprop"], "")
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in optimizer.")
    gflags.DEFINE_boolean("actively_decay_learning_rate", False, "Used in optimizer.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer("statistics_interval_steps", 100, "Print training set results at this interval.")
    gflags.DEFINE_integer("eval_interval_steps", 100, "Evaluate at this interval.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000, "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean("ckpt_on_best_dev_error", True, "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")

    # Evaluation settings
    gflags.DEFINE_boolean("expanded_eval_only_mode", False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")

    # Parse command line flags.
    FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if not FLAGS.branch_name:
        FLAGS.branch_name = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.sha:
        FLAGS.sha = os.popen('git rev-parse HEAD').read().strip()

    run(only_forward=FLAGS.expanded_eval_only_mode)
