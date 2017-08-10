import os
import json
import random
import sys
import time
import math

import gflags
import numpy as np

from spinn.util import afs_safe_logger
from spinn.util.data import SimpleProgressBar
from spinn.util.blocks import get_l2_loss, the_gpu, to_gpu
from spinn.util.misc import Accumulator, EvalReporter
from spinn.util.misc import recursively_set_device
from spinn.util.logging import stats, train_accumulate, create_log_formatter
from spinn.util.logging import eval_stats, eval_accumulate, prettyprint_trees
from spinn.util.loss import auxiliary_loss
from spinn.util.sparks import sparks, dec_str
import spinn.util.evalb as evalb
import spinn.util.logging_pb2 as pb

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from spinn.models.base import get_data_manager, get_flags, get_batch
from spinn.models.base import flag_defaults, init_model
from spinn.models.base import get_checkpoint_path, log_path
from spinn.models.base import load_data_and_embeddings


FLAGS = gflags.FLAGS


def evaluate(FLAGS, model, data_manager, eval_set, log_entry,
             logger, step, vocabulary=None, show_sample=False, eval_index=0):
    filename, dataset = eval_set

    A = Accumulator()
    index = len(log_entry.evaluation)
    eval_log = log_entry.evaluation.add()
    reporter = EvalReporter()
    tree_strs = None

    # Evaluate
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(msg="Run Eval", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    start = time.time()

    if FLAGS.model_type in ["Pyramid", "ChoiPyramid"]:
        pyramid_temperature_multiplier = FLAGS.pyramid_temperature_decay_per_10k_steps ** (
            step / 10000.0)
        if FLAGS.pyramid_temperature_cycle_length > 0.0:
            min_temp = 1e-5
            pyramid_temperature_multiplier *= (math.cos((step) /
                                                        FLAGS.pyramid_temperature_cycle_length) + 1 + min_temp) / 2
    else:
        pyramid_temperature_multiplier = None

    model.eval()
    for i, dataset_batch in enumerate(dataset):
        batch = get_batch(dataset_batch)
        eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, eval_ids = batch
        print len(eval_ids), i

        # Run model.
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
                       use_internal_parser=FLAGS.use_internal_parser,
                       validate_transitions=FLAGS.validate_transitions,
                       pyramid_temperature_multiplier=pyramid_temperature_multiplier,
                       store_parse_masks=show_sample,
                       example_lengths=eval_num_transitions_batch)

        can_sample = FLAGS.model_type in ["ChoiPyramid"] or (FLAGS.model_type == "SPINN" and FLAGS.use_internal_parser)  # TODO: Restore support in Pyramid if using.
        if show_sample and can_sample:
            tmp_samples = model.get_samples(eval_X_batch, vocabulary, only_one=not FLAGS.write_eval_report)
            tree_strs = prettyprint_trees(tmp_samples)
        if not FLAGS.write_eval_report:
            show_sample = False  # Only show one sample, regardless of the number of batches.


        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()

        # get the index of the max log-probability
        pred = logits.data.max(1, keepdim=False)[1].cpu()

        eval_accumulate(model, data_manager, A, batch)
        A.add('class_correct', pred.eq(target).sum())
        A.add('class_total', target.size(0))

        # Optionally calculate transition loss/acc.
        model.transition_loss if hasattr(model, 'transition_loss') else None

        # Update Aggregate Accuracies
        total_tokens += sum([(nt + 1) / 2 for nt in eval_num_transitions_batch.reshape(-1)])

        if FLAGS.write_eval_report:
            transitions_per_example, _ = model.spinn.get_transitions_per_example(
                    style="preds" if FLAGS.eval_report_use_preds else "given") if (FLAGS.model_type == "SPINN" and FLAGS.use_internal_parser) else (None, None)

            if model.use_sentence_pair:
                batch_size = pred.size(0)
                sent1_transitions = transitions_per_example[:batch_size] if transitions_per_example is not None else None
                sent2_transitions = transitions_per_example[batch_size:] if transitions_per_example is not None else None

                sent1_trees = tree_strs[:batch_size] if tree_strs is not None else None
                sent2_trees = tree_strs[batch_size:] if tree_strs is not None else None
            else:
                sent1_transitions = transitions_per_example if transitions_per_example is not None else None
                sent2_transitions = None

                sent1_trees = tree_strs if tree_strs is not None else None
                sent2_trees = None

            reporter.save_batch(pred, target, eval_ids, output.data.cpu().numpy(), sent1_transitions, sent2_transitions, sent1_trees, sent2_trees)

        # Print Progress
        progress_bar.step(i + 1, total=total_batches)
    progress_bar.finish()
    if tree_strs is not None:
        logger.Log('Sample: ' + tree_strs[0])

    end = time.time()
    total_time = end - start

    A.add('total_tokens', total_tokens)
    A.add('total_time', total_time)

    eval_stats(model, A, eval_log)
    eval_log.filename = filename

    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".eval_set_" + str(eval_index) + ".report")
        reporter.write_report(eval_report_path)

    eval_class_acc = eval_log.eval_class_accuracy
    eval_trans_acc = eval_log.eval_transition_accuracy

    return eval_class_acc, eval_trans_acc


def train_loop(FLAGS, data_manager, model, optimizer, trainer,
               training_data_iter, eval_iterators, logger, step, best_dev_error, vocabulary):
    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)

    # Checkpoint paths.
    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name, best=True)

    # Build log format strings.
    model.train()
    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = get_batch(
        training_data_iter.next())
    model(X_batch, transitions_batch, y_batch,
          use_internal_parser=FLAGS.use_internal_parser,
          validate_transitions=FLAGS.validate_transitions,
          pyramid_temperature_multiplier=1.0,
          example_lengths=num_transitions_batch
          )

    # Train.
    logger.Log("Training.")

    # New Training Loop
    progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

    log_entry = pb.SpinnEntry()
    for step in range(step, FLAGS.training_steps):
        model.train()
        log_entry.Clear()
        log_entry.step = step
        should_log = False

        start = time.time()

        batch = get_batch(training_data_iter.next())
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch

        total_tokens = sum([(nt + 1) / 2 for nt in num_transitions_batch.reshape(-1)])

        # Reset cached gradients.
        optimizer.zero_grad()

        if FLAGS.model_type in ["Pyramid", "ChoiPyramid"]:
            pyramid_temperature_multiplier = FLAGS.pyramid_temperature_decay_per_10k_steps ** (
                step / 10000.0)
            if FLAGS.pyramid_temperature_cycle_length > 0.0:
                min_temp = 1e-5
                pyramid_temperature_multiplier *= (math.cos((step) /
                                                            FLAGS.pyramid_temperature_cycle_length) + 1 + min_temp) / 2
        else:
            pyramid_temperature_multiplier = None

        # Run model.
        output = model(X_batch, transitions_batch, y_batch,
                       use_internal_parser=FLAGS.use_internal_parser,
                       validate_transitions=FLAGS.validate_transitions,
                       pyramid_temperature_multiplier=pyramid_temperature_multiplier,
                       example_lengths=num_transitions_batch
                       )

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(y_batch).long()

        # get the index of the max log-probability
        pred = logits.data.max(1, keepdim=False)[1].cpu()

        class_acc = pred.eq(target).sum() / float(target.size(0))

        # Calculate class loss.
        xent_loss = nn.NLLLoss()(logits, to_gpu(Variable(target, volatile=False)))

        # Optionally calculate transition loss.
        transition_loss = model.transition_loss if hasattr(model, 'transition_loss') else None

        # Extract L2 Cost
        l2_loss = get_l2_loss(model, FLAGS.l2_lambda) if FLAGS.use_l2_loss else None

        # Accumulate Total Loss Variable
        total_loss = 0.0
        total_loss += xent_loss
        if l2_loss is not None:
            total_loss += l2_loss
        if transition_loss is not None and model.optimize_transition_loss:
            total_loss += transition_loss
        aux_loss = auxiliary_loss(model)
        total_loss += aux_loss
        # Backward pass.
        total_loss.backward()

        # Hard Gradient Clipping
        clip = FLAGS.clipping_max_value
        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.clamp_(min=-clip, max=clip)

        # Learning Rate Decay
        if FLAGS.actively_decay_learning_rate:
            optimizer.lr = FLAGS.learning_rate * \
                (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

        # Gradient descent step.
        optimizer.step()

        end = time.time()

        total_time = end - start

        train_accumulate(model, data_manager, A, batch)
        A.add('class_acc', class_acc)
        A.add('total_tokens', total_tokens)
        A.add('total_time', total_time)

        if step % FLAGS.statistics_interval_steps == 0:
            A.add('xent_cost', xent_loss.data[0])
            A.add('l2_cost', l2_loss.data[0])
            stats(model, optimizer, A, step, log_entry)
            should_log = True
            progress_bar.finish()

        if step % FLAGS.sample_interval_steps == 0 and FLAGS.num_samples > 0:
            should_log = True
            model.train()
            model(X_batch, transitions_batch, y_batch,
                  use_internal_parser=FLAGS.use_internal_parser,
                  validate_transitions=FLAGS.validate_transitions,
                  pyramid_temperature_multiplier=pyramid_temperature_multiplier,
                  example_lengths=num_transitions_batch
                  )
            tr_transitions_per_example, tr_strength = model.spinn.get_transitions_per_example()

            model.eval()
            model(X_batch, transitions_batch, y_batch,
                  use_internal_parser=FLAGS.use_internal_parser,
                  validate_transitions=FLAGS.validate_transitions,
                  pyramid_temperature_multiplier=pyramid_temperature_multiplier,
                  example_lengths=num_transitions_batch
                  )
            ev_transitions_per_example, ev_strength = model.spinn.get_transitions_per_example()

            if model.use_sentence_pair and len(transitions_batch.shape) == 3:
                transitions_batch = np.concatenate([
                    transitions_batch[:, :, 0], transitions_batch[:, :, 1]], axis=0)

            # This could be done prior to running the batch for a tiny speed boost.
            t_idxs = range(FLAGS.num_samples)
            random.shuffle(t_idxs)
            t_idxs = sorted(t_idxs[:FLAGS.num_samples])
            for t_idx in t_idxs:
                log = log_entry.rl_sampling.add()
                gold = transitions_batch[t_idx]
                pred_tr = tr_transitions_per_example[t_idx]
                pred_ev = ev_transitions_per_example[t_idx]
                strength_tr = sparks([1] + tr_strength[t_idx].tolist(), dec_str)
                strength_ev = sparks([1] + ev_strength[t_idx].tolist(), dec_str)
                _, crossing = evalb.crossing(gold, pred_ev)
                log.t_idx = t_idx
                log.crossing = crossing
                log.gold_lb = "".join(map(str, gold))
                log.pred_tr = "".join(map(str, pred_tr))
                log.pred_ev = "".join(map(str, pred_ev))
                log.strg_tr = strength_tr[1:].encode('utf-8')
                log.strg_ev = strength_ev[1:].encode('utf-8')

        if step > 0 and step % FLAGS.eval_interval_steps == 0:
            should_log = True
            for index, eval_set in enumerate(eval_iterators):
                acc, tacc = evaluate(FLAGS, model, data_manager, eval_set, log_entry, logger, step,
                                     show_sample=(
                                         step %
                                         FLAGS.sample_interval_steps == 0), vocabulary=vocabulary, eval_index=index)
                if FLAGS.ckpt_on_best_dev_error and index == 0 and (
                        1 - acc) < 0.99 * best_dev_error and step > FLAGS.ckpt_step:
                    best_dev_error = 1 - acc
                    logger.Log("Checkpointing with new best dev accuracy of %f" % acc)  # TODO: This mixes information across dev sets. Fix.
                    trainer.save(best_checkpoint_path, step, best_dev_error)
            progress_bar.reset()

        if step > FLAGS.ckpt_step and step % FLAGS.ckpt_interval_steps == 0:
            should_log = True
            logger.Log("Checkpointing.")
            trainer.save(standard_checkpoint_path, step, best_dev_error)

        if should_log:
            logger.LogEntry(log_entry)

        progress_bar.step(i=(step % FLAGS.statistics_interval_steps) + 1,
                          total=FLAGS.statistics_interval_steps)


def run(only_forward=False):
    logger = afs_safe_logger.ProtoLogger(log_path(FLAGS),
                                         print_formatter=create_log_formatter(True, False),
                                         write_proto=FLAGS.write_proto_to_log)
    header = pb.SpinnHeader()

    data_manager = get_data_manager(FLAGS.data_type)

    logger.Log("Flag Values:\n" +
               json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))
    flags_dict = sorted(list(FLAGS.FlagValuesDict().items()))
    for k, v in flags_dict:
        flag = header.flags.add()
        flag.key = k
        flag.value = str(v)

    # Get Data and Embeddings
    vocabulary, initial_embeddings, training_data_iter, eval_iterators = \
        load_data_and_embeddings(FLAGS, data_manager, logger,
                                 FLAGS.training_data_path, FLAGS.eval_data_path)

    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(set(data_manager.LABEL_MAP.values()))

    model, optimizer, trainer = init_model(
        FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager, header)

    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, FLAGS.experiment_name, best=True)

    # Load checkpoint if available.
    if FLAGS.load_best and os.path.isfile(best_checkpoint_path):
        logger.Log("Found best checkpoint, restoring.")
        step, best_dev_error = trainer.load(best_checkpoint_path)
        logger.Log(
            "Resuming at step: {} with best dev accuracy: {}".format(
                step, 1. - best_dev_error))
    elif os.path.isfile(standard_checkpoint_path):
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error = trainer.load(standard_checkpoint_path)
        logger.Log(
            "Resuming at step: {} with best dev accuracy: {}".format(
                step, 1. - best_dev_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0
    header.start_step = step
    header.start_time = int(time.time())

    # GPU support.
    the_gpu.gpu = FLAGS.gpu
    if FLAGS.gpu >= 0:
        model.cuda()
    else:
        model.cpu()
    recursively_set_device(optimizer.state_dict(), FLAGS.gpu)

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)

    # Do an evaluation-only run.
    logger.LogHeader(header)  # Start log_entry logging.
    if only_forward:
        log_entry = pb.SpinnEntry()
        for index, eval_set in enumerate(eval_iterators):
            log_entry.Clear()
            evaluate(FLAGS, model, data_manager, eval_set, log_entry, logger, step, vocabulary, show_sample=True, eval_index=index)
            print(log_entry)
            logger.LogEntry(log_entry)
    else:
        train_loop(FLAGS, data_manager, model, optimizer, trainer,
                   training_data_iter, eval_iterators, logger, step, best_dev_error, vocabulary)


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS)

    if FLAGS.model_type == "RLSPINN":
        raise Exception(
            "Please use rl_classifier.py instead of supervised_classifier.py for RLSPINN.")

    run(only_forward=FLAGS.expanded_eval_only_mode)
