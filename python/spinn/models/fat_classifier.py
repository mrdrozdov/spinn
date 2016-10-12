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

from functools import partial
import os
import pprint
import sys

import gflags
import numpy as np

from spinn import afs_safe_logger
from spinn import util
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.util import chainer_blocks as CB

import spinn.fat_stack
import spinn.plain_rnn_chainer
import spinn.cbow


FLAGS = gflags.FLAGS


def build_sentence_pair_model(cls, vocab_size, seq_length, tokens, transitions,
                     num_classes, training_mode, ground_truth_transitions_visible, vs,
                     initial_embeddings=None, project_embeddings=False, ss_mask_gen=None, ss_prob=0.0):
    """
    Construct a classifier which makes use of some hard-stack model.

    Args:
      cls: Hard stack class to use (from e.g. `spinn.fat_stack`)
      vocab_size:
      seq_length: Length of each sequence provided to the stack model
      num_classes: Number of output classes
      training_mode: A Theano scalar indicating whether to act as a training model
        with dropout (1.0) or to act as an eval model with rescaling (0.0).
    """


    # Prepare layer which performs stack element composition.
    if cls is spinn.plain_rnn_chainer.RNN:
        compose_network = CB.LSTM
    elif cls is spinn.fat_stack.TransitionModel:
        compose_network = CB.LSTM
    else:
        raise AssertionError("Need to specify an implemented model.")

    classifier_model = cls(FLAGS.model_dim, FLAGS.word_embedding_dim, vocab_size, compose_network,
        keep_rate=FLAGS.embedding_keep_rate,
        seq_length=seq_length,
        num_classes=num_classes,
        mlp_dim=1024,
        initial_embeddings=initial_embeddings,
        use_sentence_pair=True,
        gpu=FLAGS.gpu,
        )

    return classifier_model


def evaluate(classifier_model, eval_set, logger, step):
    # Evaluate
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    for (X_batch, transitions_batch, y_batch, num_transitions_batch) in eval_set[1]:
        # Calculate Local Accuracies
        classifier_model.forward(X_batch, y_batch, train=False, predict=False)
        acc_value = float(classifier_model.model.accuracy.data)
        action_acc_value = 0.0

        # Update Aggregate Accuracies
        acc_accum += acc_value
        action_acc_accum += action_acc_value
        eval_batches += 1.0
    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
              (step, acc_accum / eval_batches, action_acc_accum / eval_batches, eval_set[0]))
    return acc_accum / eval_batches


def evaluate_expanded(eval_fn, eval_set, eval_path, logger, step, sentence_pair_data, ind_to_word, predict_transitions):
    """
    Write the  gold parses and predicted parses in the files <eval_out_path>.gld and <eval_out_path>.tst
    respectively. These files can be given as inputs to Evalb to evaluate parsing performance -

        evalb -p evalb_spinn.prm <eval_out_path>.gld  <eval_out_path>.tst

    TODO(SB): Set up for RNN and Model0 on non-sentence-pair data; port support to classifier.py.
    """
    # TODO: Prune out redundant code, make usable on Model0 as well.
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    eval_gold_path = eval_path + ".gld"
    eval_out_path = eval_path + ".tst"
    eval_lbl_path = eval_path + ".lbl"
    with open(eval_gold_path, "w") as eval_gold, open(eval_out_path, "w") as eval_out:
        if FLAGS.write_predicted_label:
            label_out = open(eval_lbl_path, "w")
        if sentence_pair_data:
            for (eval_X_batch, eval_transitions_batch, eval_y_batch,
                    eval_num_transitions_batch) in eval_set[1]:
                acc_value, action_acc_value, sem_logit_values, logits_pred_hyp, logits_pred_prem = eval_fn(
                    eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch,
                    0.0,  # Eval mode: Don't apply dropout.
                    int(FLAGS.allow_gt_transitions_in_eval),  # Allow GT transitions to be used according to flag.
                    float(FLAGS.allow_gt_transitions_in_eval)) # adjust visibility of GT

                acc_accum += acc_value
                action_acc_accum += action_acc_value
                eval_batches += 1.0

                # write each predicted transition to file
                for orig_transitions, pred_logit_hyp, pred_logit_prem, tokens, true_class, example_sem_logits \
                        in zip(eval_transitions_batch, logits_pred_hyp,
                               logits_pred_prem, eval_X_batch, eval_y_batch, sem_logit_values):
                    if predict_transitions:
                        orig_hyp_transitions, orig_prem_transitions = orig_transitions.T
                        pred_hyp_transitions = pred_logit_hyp.argmax(axis=1)
                        pred_prem_transitions = pred_logit_prem.argmax(axis=1)
                    else:
                        orig_hyp_transitions = orig_prem_transitions = pred_hyp_transitions = pred_prem_transitions = None

                    hyp_tokens, prem_tokens = tokens.T
                    hyp_words = [ind_to_word[t] for t in hyp_tokens]
                    prem_words = [ind_to_word[t] for t in prem_tokens]
                    eval_gold.write(util.TransitionsToParse(orig_hyp_transitions, hyp_words) + "\n")
                    eval_out.write(util.TransitionsToParse(pred_hyp_transitions, hyp_words) + "\n")
                    eval_gold.write(util.TransitionsToParse(orig_prem_transitions, prem_words) + "\n")
                    eval_out.write(util.TransitionsToParse(pred_prem_transitions, prem_words) + "\n")

                    predicted_class = np.argmax(example_sem_logits)
                    exp_logit_values = np.exp(example_sem_logits)
                    class_probs = exp_logit_values / np.sum(exp_logit_values)
                    class_probs_repr = "\t".join(map(lambda p : "%.8f" % (p,), class_probs))
                    if FLAGS.write_predicted_label:
                        label_out.write(str(true_class == predicted_class) + "\t" + str(true_class)
                                  + "\t" + str(predicted_class) + "\t" + class_probs_repr + "\n")
        else:
            for (eval_X_batch, eval_transitions_batch, eval_y_batch,
                 eval_num_transitions_batch) in eval_set[1]:
                acc_value, action_acc_value, sem_logit_values, logits_pred = eval_fn(
                    eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch,
                    0.0,  # Eval mode: Don't apply dropout.
                    int(FLAGS.allow_gt_transitions_in_eval),  # Allow GT transitions to be used according to flag.
                    float(FLAGS.allow_gt_transitions_in_eval)) # adjust visibility of GT

                acc_accum += acc_value
                action_acc_accum += action_acc_value
                eval_batches += 1.0

                # write each predicted transition to file
                for orig_transitions, pred_logit, tokens, true_class, example_sem_logits \
                    in zip(eval_transitions_batch, logits_pred, eval_X_batch, eval_y_batch, sem_logit_values):
                    words = [ind_to_word[t] for t in tokens]
                    eval_gold.write(util.TransitionsToParse(orig_transitions, words) + "\n")
                    eval_out.write(util.TransitionsToParse(pred_logit.argmax(axis=1), words) + "\n")

                    predicted_class = np.argmax(example_sem_logits)
                    exp_logit_values = np.exp(example_sem_logits)
                    class_probs = exp_logit_values / np.sum(exp_logit_values)
                    class_probs_repr = "\t".join(map(lambda p : "%.3f" % (p,), class_probs))
                    if FLAGS.write_predicted_label:
                        label_out.write(str(true_class == predicted_class) + "\t" + str(true_class)
                                    + "\t" + str(predicted_class) + "\t" + class_probs_repr + "\n")

    logger.Log("Written gold parses in %s" % (eval_gold_path))
    logger.Log("Written predicted parses in %s" % (eval_out_path))
    if FLAGS.write_predicted_label:
        logger.Log("Written predicted labels in %s" % (eval_lbl_path))
        label_out.close()
    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
               (step, acc_accum / eval_batches, action_acc_accum / eval_batches, eval_set[0]))


def run(only_forward=False):
    logger = afs_safe_logger.Logger(os.path.join(FLAGS.log_path, FLAGS.experiment_name) + ".log")

    if FLAGS.data_type == "bl":
        data_manager = load_boolean_data
    elif FLAGS.data_type == "sst":
        data_manager = load_sst_data
    elif FLAGS.data_type == "snli":
        data_manager = load_snli_data
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
        initial_embeddings = util.LoadEmbeddingsFromASCII(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data, vocabulary, FLAGS.seq_length, data_manager, eval_mode=False, logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW")
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size)

    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        e_X, e_transitions, e_y, e_num_transitions = util.PreprocessDataset(
            raw_eval_set, vocabulary, FLAGS.seq_length, data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW")
        eval_iterators.append((filename,
            util.MakeEvalIterator((e_X, e_transitions, e_y, e_num_transitions), FLAGS.batch_size)))

    # Set up the placeholders.

    logger.Log("Building model.")

    if FLAGS.model_type == "CBOW":
        model_cls = spinn.cbow.CBOW
    elif FLAGS.model_type == "RNN":
        model_cls = spinn.plain_rnn_chainer.RNN
    elif FLAGS.model_type == "SPINN":
        # model_cls = getattr(spinn.fat_stack, FLAGS.model_type)
        model_cls = spinn.fat_stack.TransitionModel
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)

    # Generator of mask for scheduled sampling
    numpy_random = np.random.RandomState(1234)

    # Training step number

    # Dirty Hack: Using 
    y = None
    X = None
    lr = None
    transitions = None
    num_transitions = None
    training_mode = None
    ground_truth_transitions_visible = None
    vs = None
    ss_mask_gen = None
    ss_prob = None

    predicted_transitions = None
    predicted_premise_transitions = None
    predicted_hypothesis_transitions = None
    logits = None

    if data_manager.SENTENCE_PAIR_DATA:
        classifier_model = build_sentence_pair_model(
            model_cls, len(vocabulary), FLAGS.seq_length,
            X, transitions, len(data_manager.LABEL_MAP), training_mode, ground_truth_transitions_visible, vs,
            initial_embeddings=initial_embeddings, project_embeddings=(not train_embeddings),
            ss_mask_gen=ss_mask_gen,
            ss_prob=ss_prob)
    else:
        raise Exception("Single sentence model not implemented.")

    if ".ckpt" in FLAGS.ckpt_path:
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + ".ckpt")
    if os.path.isfile(checkpoint_path):
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = vs.load_checkpoint(checkpoint_path, num_extra_vars=2,
                                                  skip_saved_unsavables=FLAGS.skip_saved_unsavables)
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    # Do an evaluation-only run.
    if only_forward:
        raise Exception("Not implemented for chainer.")
    else:
         # Train
        logger.Log("Training.")
        
        classifier_model.init_optimizer(
            clip=FLAGS.clipping_max_value, decay=FLAGS.l2_lambda,
            lr=FLAGS.learning_rate,
            )

        # New Training Loop
        for step in range(step, FLAGS.training_steps):
            if step > 0 and step % FLAGS.eval_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(classifier_model, eval_set, logger, step)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > 1000:
                        best_dev_error = 1 - acc
                        logger.Log("[TODO: NOT IMPLEMENTED] Checkpointing with new best dev accuracy of %f" % acc)
            X_batch, transitions_batch, y_batch, num_transitions_batch = training_data_iter.next()
            learning_rate = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

            # Reset hidden states of RNN(s), and reset cached gradients.
            classifier_model.model.cleargrads()

            # Calculate loss and update parameters.
            ret = classifier_model.forward({
                "sentences": X_batch,
                "transitions": transitions_batch,
                }, y_batch, train=True, predict=False)
            y, loss, preds = ret

            # Boilerplate for calculating loss.
            xent_cost_val = loss.data
            transition_cost_val = 0.0

            total_cost_val = xent_cost_val + transition_cost_val
            loss.backward()
            classifier_model.update()

            # Accumulate accuracy for current interval.
            action_acc_val = 0.0
            acc_val = float(classifier_model.model.accuracy.data)

            if step % FLAGS.statistics_interval_steps == 0:
                logger.Log(
                    "Step: %i\tAcc: %f\t%f\tCost: %5f %5f %5f %s"
                    % (step, acc_val, action_acc_val, total_cost_val, xent_cost_val, transition_cost_val,
                       "l2-not-exposed"))


if __name__ == '__main__':
    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "experiment", "")

    # Data types.
    gflags.DEFINE_enum("data_type", "bl", ["bl", "sst", "snli"],
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
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", 30, "")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN",
                       ["CBOW", "RNN", "SPINN"],
                       "")
    gflags.DEFINE_boolean("allow_gt_transitions_in_eval", False,
        "Whether to use ground truth transitions in evaluation when appropriate "
        "(i.e., in Model 1 and Model 2S.)")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")

    gflags.DEFINE_integer("tracking_lstm_hidden_dim", 4, "")
    gflags.DEFINE_boolean("use_tracking_lstm", True,
                          "Whether to use LSTM in the tracking unit")
    gflags.DEFINE_boolean("predict_use_cell", False,
                          "For models which predict parser actions, use "
                          "both the tracking LSTM hidden and cell values as "
                          "input to the prediction layer")
    gflags.DEFINE_boolean("context_sensitive_shift", False,
        "Use LSTM hidden state and word embedding to determine the vector to be pushed")
    gflags.DEFINE_boolean("context_sensitive_use_relu", False,
        "Use ReLU Layer to combine embedding and tracking unit hidden state")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.5,
        "Used for dropout in the semantic task classifier.")
    gflags.DEFINE_float("embedding_keep_rate", 0.5,
        "Used for dropout on transformed embeddings.")
    gflags.DEFINE_boolean("lstm_composition", True, "")
    gflags.DEFINE_enum("classifier_type", "MLP", ["MLP", "Highway", "ResNet"], "")
    gflags.DEFINE_integer("resnet_unit_depth", 2, "")
    # gflags.DEFINE_integer("num_composition_layers", 1, "")
    gflags.DEFINE_integer("num_sentence_pair_combination_layers", 2, "")
    gflags.DEFINE_integer("sentence_pair_combination_layer_dim", 1024, "")
    gflags.DEFINE_float("scheduled_sampling_exponent_base", 0.99,
        "Used for scheduled sampling, with probability of Model 1 over Model 2 being base^#training_steps")
    gflags.DEFINE_boolean("use_difference_feature", True,
        "Supply the sentence pair classifier with sentence difference features.")
    gflags.DEFINE_boolean("use_product_feature", True,
        "Supply the sentence pair classifier with sentence product features.")
    gflags.DEFINE_boolean("connect_tracking_comp", True,
        "Connect tracking unit and composition unit. Can only be true if using LSTM in both units.")
    gflags.DEFINE_boolean("initialize_hyp_tracking_state", False,
        "Initialize the c state of the tracking unit of hypothesis model with the final"
        "tracking unit c state of the premise model.")
    gflags.DEFINE_boolean("use_gru", False,
                          "Use GRU units instead of LSTM units.")

    # Optimization settings.
    gflags.DEFINE_integer("training_steps", 500000, "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in RMSProp.")
    gflags.DEFINE_float("learning_rate_decay_per_10k_steps", 0.75, "Used in RMSProp.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float("init_range", 0.005, "Mainly used for softmax parameters. Range for uniform random init.")
    gflags.DEFINE_float("transition_cost_scale", 1.0, "Multiplied by the transition cost.")

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
    gflags.DEFINE_string("eval_output_paths", None,
        "Used when expanded_eval_only_mode is set. The number of supplied paths should be same"
        "as the number of eval sets.")
    gflags.DEFINE_boolean("write_predicted_label", False,
        "Write the predicted labels in a <eval_output_name>.lbl file.")
    gflags.DEFINE_boolean("skip_saved_unsavables", False,
        "Assume that variables marked as not savable will appear in checkpoints anyway, and "
        "skip them when loading. This should be used only when loading old checkpoints.")

    # Parse command line flags.
    FLAGS(sys.argv)

    run(only_forward=FLAGS.expanded_eval_only_mode)
