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
import time

import gflags
import numpy as np

from spinn import afs_safe_logger
from spinn import util
from spinn.data.boolean import load_boolean_data
from spinn.data.sst import load_sst_data
from spinn.data.snli import load_snli_data
from spinn.util.data import SimpleProgressBar
from spinn.util.chainer_blocks import gradient_check

import spinn.fat_stack
import spinn.plain_rnn_chainer
import spinn.cbow
import spinn.nti


FLAGS = gflags.FLAGS


def build_sentence_pair_model(model_cls, trainer_cls, model_dim, word_embedding_dim,
                              seq_length, num_classes, initial_embeddings,
                              keep_rate, gpu):
    model = model_cls(model_dim, word_embedding_dim,
             seq_length, initial_embeddings, num_classes, mlp_dim=1024,
             keep_rate=keep_rate,
             gpu=gpu,
             tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
             transition_weight=FLAGS.transition_weight,
             use_tracking_lstm=FLAGS.use_tracking_lstm,
             use_shift_composition=FLAGS.use_shift_composition,
             make_logits=FLAGS.make_logits,
             use_history=FLAGS.use_history,
             save_stack=FLAGS.save_stack,
            )

    classifier_trainer = trainer_cls(model, model_dim, word_embedding_dim,
        keep_rate=keep_rate,
        seq_length=seq_length,
        num_classes=num_classes,
        mlp_dim=1024,
        initial_embeddings=initial_embeddings,
        use_sentence_pair=True,
        gpu=gpu,
        )

    return classifier_trainer


def evaluate(classifier_trainer, eval_set, logger, step):
    # Evaluate
    acc_accum = 0.0
    action_acc_accum = 0.0
    eval_batches = 0.0
    total_batches = len(eval_set[1])
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    for i, (eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch) in enumerate(eval_set[1]):
        # Calculate Local Accuracies
        ret = classifier_trainer.forward({
            "sentences": eval_X_batch,
            "transitions": eval_transitions_batch,
            }, eval_y_batch, train=False, predict=False)
        y, loss, class_loss, transition_acc, transition_loss = ret
        acc_value = float(classifier_trainer.model.accuracy.data)
        action_acc_value = transition_acc

        # Update Aggregate Accuracies
        acc_accum += acc_value
        action_acc_accum += action_acc_value
        eval_batches += 1.0

        # Print Progress
        progress_bar.step(i+1, total=total_batches)
    progress_bar.finish()
    logger.Log("Step: %i\tEval acc: %f\t %f\t%s" %
              (step, acc_accum / eval_batches, action_acc_accum / eval_batches, eval_set[0]))
    return acc_accum / eval_batches


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
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano)

    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        e_X, e_transitions, e_y, e_num_transitions = util.PreprocessDataset(
            raw_eval_set, vocabulary, FLAGS.seq_length, data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW")
        eval_iterators.append((filename,
            util.MakeEvalIterator((e_X, e_transitions, e_y, e_num_transitions),
                FLAGS.batch_size, FLAGS.eval_data_limit)))

    # Set up the placeholders.

    logger.Log("Building model.")

    if FLAGS.model_type == "CBOW":
        model_module = spinn.cbow
    elif FLAGS.model_type == "RNN":
        model_module = spinn.plain_rnn_chainer
    elif FLAGS.model_type == "NTI":
        model_module = spinn.nti
    elif FLAGS.model_type == "SPINN":
        model_module = spinn.fat_stack
    else:
        raise Exception("Requested unimplemented model type %s" % FLAGS.model_type)


    if data_manager.SENTENCE_PAIR_DATA:
        if hasattr(model_module, 'SentencePairTrainer') and hasattr(model_module, 'SentencePairModel'):
            trainer_cls = model_module.SentencePairTrainer
            model_cls = model_module.SentencePairModel
        else:
            raise Exception("Unimplemented for model type %s" % FLAGS.model_type)

        num_classes = len(data_manager.LABEL_MAP)
        classifier_trainer = build_sentence_pair_model(model_cls, trainer_cls,
                              FLAGS.model_dim, FLAGS.word_embedding_dim,
                              FLAGS.seq_length, num_classes, initial_embeddings,
                              FLAGS.embedding_keep_rate, FLAGS.gpu)
    else:
        if hasattr(model_module, 'SentencePairTrainer') and hasattr(model_module, 'SentencePairModel'):
            trainer_cls = model_module.SentenceTrainer
            model_cls = model_module.SentenceModel
        else:
            raise Exception("Unimplemented for model type %s" % FLAGS.model_type)

        num_classes = len(data_manager.LABEL_MAP)
        classifier_trainer = build_sentence_pair_model(model_cls, trainer_cls,
                              FLAGS.model_dim, FLAGS.word_embedding_dim,
                              FLAGS.seq_length, num_classes, initial_embeddings,
                              FLAGS.embedding_keep_rate, FLAGS.gpu)

    if ".ckpt" in FLAGS.ckpt_path:
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + ".ckpt")
    if os.path.isfile(checkpoint_path):
        # TODO: Check that resuming works fine with tf summaries.
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = classifier_trainer.load(checkpoint_path)
        logger.Log("Resuming at step: {} with best dev accuracy: {}".format(step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0

    if FLAGS.write_summaries:
        from spinn.tf_logger import TFLogger
        train_summary_logger = TFLogger(summary_dir=os.path.join(FLAGS.summary_dir, FLAGS.experiment_name, 'train'))
        dev_summary_logger = TFLogger(summary_dir=os.path.join(FLAGS.summary_dir, FLAGS.experiment_name, 'dev'))

    # Do an evaluation-only run.
    if only_forward:
        raise Exception("Not implemented for chainer.")
    else:
         # Train
        logger.Log("Training.")

        classifier_trainer.init_optimizer(
            clip=FLAGS.clipping_max_value, decay=FLAGS.l2_lambda,
            lr=FLAGS.learning_rate,
            )

        # New Training Loop
        progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
        avg_class_acc = 0
        avg_trans_acc = 0
        for step in range(step, FLAGS.training_steps):
            X_batch, transitions_batch, y_batch, num_transitions_batch = training_data_iter.next()
            learning_rate = FLAGS.learning_rate * (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

            # Reset cached gradients.
            classifier_trainer.optimizer.zero_grads()

            # Calculate loss and update parameters.
            ret = classifier_trainer.forward({
                "sentences": X_batch,
                "transitions": transitions_batch,
                }, y_batch, train=True, predict=False)
            y, loss, class_acc, transition_acc, transition_loss = ret

            # Boilerplate for calculating loss.
            xent_cost_val = loss.data
            transition_cost_val = transition_loss.data if transition_loss is not None else 0.0
            avg_trans_acc += transition_acc
            avg_class_acc += class_acc

            if FLAGS.show_intermediate_stats and step % 5 == 0 and step % FLAGS.statistics_interval_steps > 0:
                print("Accuracies so far : ", avg_class_acc / (step % FLAGS.statistics_interval_steps), avg_trans_acc / (step % FLAGS.statistics_interval_steps))

            total_cost_val = xent_cost_val + transition_cost_val
            loss.backward()
            if hasattr(transition_loss, 'backward'):
              transition_loss.backward()

            if FLAGS.gradient_check:
                def get_loss():
                    _, check_loss, _, _ = classifier_trainer.forward({
                    "sentences": X_batch,
                    "transitions": transitions_batch,
                    }, y_batch, train=True, predict=False)
                    return check_loss
                gradient_check(classifier_trainer.model, get_loss)

            try:
                classifier_trainer.update()
            except:
                import ipdb; ipdb.set_trace()
                pass

            # Accumulate accuracy for current interval.
            action_acc_val = 0.0
            acc_val = float(classifier_trainer.model.accuracy.data)

            if FLAGS.write_summaries:
                train_summary_logger.log(step=step, loss=total_cost_val, accuracy=acc_val)

            progress_bar.step(
                i=max(0, step-1) % FLAGS.statistics_interval_steps + 1,
                total=FLAGS.statistics_interval_steps)

            if step % FLAGS.statistics_interval_steps == 0:
                progress_bar.finish()
                avg_class_acc /= FLAGS.statistics_interval_steps
                avg_trans_acc /= FLAGS.statistics_interval_steps
                logger.Log(
                    "Step: %i\tAcc: %f\t%f\tCost: %5f %5f %5f %s"
                    % (step, avg_class_acc, avg_trans_acc, total_cost_val, xent_cost_val, transition_cost_val,
                       "l2-not-exposed"))
                avg_trans_acc = 0
                avg_class_acc = 0

            if step > 0 and step % FLAGS.eval_interval_steps == 0:
                for index, eval_set in enumerate(eval_iterators):
                    acc = evaluate(classifier_trainer, eval_set, logger, step)
                    if FLAGS.ckpt_on_best_dev_error and index == 0 and (1 - acc) < 0.99 * best_dev_error and step > 1000:
                        best_dev_error = 1 - acc
                        logger.Log("Checkpointing with new best dev accuracy of %f" % acc)
                        classifier_trainer.save(checkpoint_path, step, best_dev_error)
                    if FLAGS.write_summaries:
                        dev_summary_logger.log(step=step, loss=0.0, accuracy=acc)
                progress_bar.reset()

            if FLAGS.profile and step >= FLAGS.profile_steps:
                break


if __name__ == '__main__':
    # Debug settings.
    gflags.DEFINE_bool("debug", True, "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool("gradient_check", False, "Randomly check that gradients match estimates.")
    gflags.DEFINE_bool("profile", False, "Set to True to quit after a few batches.")
    gflags.DEFINE_bool("write_summaries", False, "Toggle which controls whether summaries are written.")
    gflags.DEFINE_bool("show_progress_bar", True, "Turn this off when running experiments on HPC.")
    gflags.DEFINE_bool("show_intermediate_stats", False, "Print stats more frequently than regular interval."
                                                         "Mostly to retain timing with progress bar")
    gflags.DEFINE_integer("profile_steps", 3, "Specify how many steps to profile.")

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
    gflags.DEFINE_string("summary_dir", ".", "A directory in which to write summaries.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string("eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 30, "")
    gflags.DEFINE_integer("eval_seq_length", 30, "")
    gflags.DEFINE_boolean("smart_batching", True, "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer("eval_data_limit", -1, "Truncate evaluation set. -1 indicates no truncation.")
    gflags.DEFINE_string("embedding_data_path", None,
        "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum("model_type", "RNN",
                       ["CBOW", "RNN", "SPINN", "NTI"],
                       "")
    gflags.DEFINE_boolean("allow_gt_transitions_in_eval", False,
        "Whether to use ground truth transitions in evaluation when appropriate "
        "(i.e., in Model 1 and Model 2S.)")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")

    gflags.DEFINE_float("transition_weight", None, "")
    gflags.DEFINE_integer("tracking_lstm_hidden_dim", 4, "")
    gflags.DEFINE_boolean("use_shift_composition", True, "")
    gflags.DEFINE_boolean("use_history", False, "")
    gflags.DEFINE_boolean("save_stack", False, "")
    gflags.DEFINE_boolean("use_tracking_lstm", True,
                          "Whether to use LSTM in the tracking unit")
    gflags.DEFINE_boolean("make_logits", False, "Predict parser actions.")
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

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
            )

    if not FLAGS.debug:
        chainer.set_debug(False)
        os.environ['CHAINER_TYPE_CHECK'] = '0'

    run(only_forward=FLAGS.expanded_eval_only_mode)
