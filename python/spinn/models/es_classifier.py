"""
NOTE: Work in progress.

Classifier script to use evolution strategy to train the parser in an unsupervised manner.

TODO:
    - Fix only forward pass.
    - Write mirrored sampling: commented out bits in generate_seeds_and_models lays groundwork.
    - Impose stopping criteria?
"""


import os
import json
import random
import sys
import time
import glob
from shutil import copyfile

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
import torch.multiprocessing as mp

from spinn.models.base import get_data_manager, get_flags, get_batch
from spinn.models.base import flag_defaults, init_model
from spinn.models.base import get_checkpoint_path, log_path
from spinn.models.base import load_data_and_embeddings


FLAGS = gflags.FLAGS

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

def evaluate(FLAGS, model, data_manager, eval_set, log_entry, step, vocabulary=None, show_sample=False, eval_index=0):
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

    model.eval()
    for i, dataset_batch in enumerate(dataset):
        batch = get_batch(dataset_batch)
        eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, eval_ids = batch

        # Run model.
        """
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
                       use_internal_parser=FLAGS.use_internal_parser,
                       validate_transitions=FLAGS.validate_transitions)
        """
        output = model(eval_X_batch, eval_transitions_batch, eval_y_batch,
                       use_internal_parser=FLAGS.use_internal_parser,
                       validate_transitions=FLAGS.validate_transitions,
                       store_parse_masks=show_sample,
                       example_lengths=eval_num_transitions_batch)

        can_sample = (FLAGS.model_type == "SPINN" and FLAGS.use_internal_parser)
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

        """
        if FLAGS.write_eval_report:
            reporter_args = [pred, target, eval_ids, output.data.cpu().numpy()]
            if hasattr(model, 'transition_loss'):
                transitions_per_example, _ = model.spinn.get_transitions_per_example(
                    style="preds" if FLAGS.eval_report_use_preds else "given")
                if model.use_sentence_pair:
                    batch_size = pred.size(0)
                    sent1_transitions = transitions_per_example[:batch_size]
                    sent2_transitions = transitions_per_example[batch_size:]
                    reporter_args.append(sent1_transitions)
                    reporter_args.append(sent2_transitions)
                else:
                    reporter_args.append(transitions_per_example)
            reporter.save_batch(*reporter_args)
        """

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

    end = time.time()
    total_time = end - start

    A.add('total_tokens', total_tokens)
    A.add('total_time', total_time)

    eval_stats(model, A, eval_log)
    eval_log.filename = filename

    """
    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".report")
        reporter.write_report(eval_report_path)
    """

    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".eval_set_" + str(eval_index) + ".report")
        reporter.write_report(eval_report_path)

    eval_class_acc = eval_log.eval_class_accuracy
    eval_trans_acc = eval_log.eval_transition_accuracy

    return eval_class_acc, eval_trans_acc


def train_loop(FLAGS, data_manager, model, optimizer, trainer,
               training_data_iter, eval_iterators, logger, true_step, best_dev_error, perturbation_id, ev_step, header, root_id, vocabulary):
    perturbation_name = FLAGS.experiment_name + "_p" + str(perturbation_id)
    root_name = FLAGS.experiment_name + "_p" + str(root_id)
    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)

    header.start_step = true_step
    header.start_time = int(time.time())
    #header.model_label = perturbation_name

    # Checkpoint paths.
    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, perturbation_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, perturbation_name, best=True)

    # Build log format strings.
    model.train()
    X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = get_batch(
        training_data_iter.next())

    model(X_batch, transitions_batch, y_batch,
          use_internal_parser=FLAGS.use_internal_parser,
          validate_transitions=FLAGS.validate_transitions)

    # Train.
    logger.Log("Training perturbation %s" % perturbation_id)

    # New Training Loop
    progress_bar = SimpleProgressBar(msg="Training", bar_length=60, enabled=FLAGS.show_progress_bar)
    progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

    log_entry = pb.SpinnEntry()
    for step in range(FLAGS.es_episode_length):
        true_step += 1
        model.train()
        log_entry.Clear()
        log_entry.step = true_step
        log_entry.model_label = perturbation_name
        log_entry.root_label = root_name
        should_log = False

        start = time.time()

        batch = get_batch(training_data_iter.next())
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids = batch

        total_tokens = sum([(nt + 1) / 2 for nt in num_transitions_batch.reshape(-1)])

        # Reset cached gradients.
        optimizer.zero_grad()

        # Run model.
        output = model(X_batch, transitions_batch, y_batch,
                       use_internal_parser=FLAGS.use_internal_parser,
                       validate_transitions=FLAGS.validate_transitions
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
                (FLAGS.learning_rate_decay_per_10k_steps ** (true_step / 10000.0))

        # Gradient descent step.
        optimizer.step()

        end = time.time()

        total_time = end - start

        train_accumulate(model, data_manager, A, batch)
        A.add('class_acc', class_acc)
        A.add('total_tokens', total_tokens)
        A.add('total_time', total_time)

        if true_step % FLAGS.statistics_interval_steps == 0:
            progress_bar.step(i=FLAGS.statistics_interval_steps,
                              total=FLAGS.statistics_interval_steps)
            progress_bar.finish()

            A.add('xent_cost', xent_loss.data[0])
            A.add('l2_cost', l2_loss.data[0])
            stats(model, optimizer, A, true_step, log_entry)
            should_log = True

        if true_step % FLAGS.sample_interval_steps == 0 and FLAGS.num_samples > 0:
            should_log = True
            model.train()
            model(X_batch, transitions_batch, y_batch,
                  use_internal_parser=FLAGS.use_internal_parser,
                  validate_transitions=FLAGS.validate_transitions
                  )
            tr_transitions_per_example, tr_strength = model.spinn.get_transitions_per_example()

            model.eval()
            model(X_batch, transitions_batch, y_batch,
                  use_internal_parser=FLAGS.use_internal_parser,
                  validate_transitions=FLAGS.validate_transitions
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

        if true_step > 0 and true_step % FLAGS.eval_interval_steps == 0:
            should_log = True
            for index, eval_set in enumerate(eval_iterators):
                acc, tacc = evaluate(FLAGS, model, data_manager, eval_set,
                                         log_entry, true_step, vocabulary,
                                         show_sample=(true_step %FLAGS.sample_interval_steps == 0),
                                         eval_index=index)
                if FLAGS.ckpt_on_best_dev_error and index == 0 and \
                    (1 - acc) < 0.99 * best_dev_error and \
                    true_step > FLAGS.ckpt_step:
                    best_dev_error = 1 - acc
                    logger.Log("Checkpointing with new best dev accuracy of %f" % acc) # TODO: This mixes information across dev sets. Fix.
                    trainer.save(best_checkpoint_path, true_step, best_dev_error, ev_step)
            progress_bar.reset()

        if true_step > FLAGS.ckpt_step and true_step % FLAGS.ckpt_interval_steps == 0:
            should_log = True
            logger.Log("Checkpointing.")
            trainer.save(standard_checkpoint_path, true_step, best_dev_error, ev_step)

        if should_log:
            logger.LogEntry(log_entry)

        progress_bar.step(i=(true_step % FLAGS.statistics_interval_steps) + 1,
                          total=FLAGS.statistics_interval_steps)

    if os.path.exists(best_checkpoint_path):
        return ev_step, true_step, perturbation_id, best_dev_error
    else:
        return ev_step, true_step, perturbation_id, (1 - acc)


def rollout(queue, perturbed_model, FLAGS, data_manager,
            model, optimizer, trainer, training_data_iter,
            eval_iterators, logger, true_step, best_dev_error, perturbation_id, ev_step, header, root_id, vocabulary):
    """
    Train each episode
    """
    perturbation_name = FLAGS.experiment_name + "_p" + str(perturbation_id)
    root_name = FLAGS.experiment_name + "_p" + str(root_id)
    logger.Log("Model name is %s" % perturbation_name)
    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, perturbation_name)
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, perturbation_name, best=True)
    root_best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, root_name, best=True)
    if os.path.exists(root_best_checkpoint_path) and root_best_checkpoint_path != best_checkpoint_path:
        copyfile(root_best_checkpoint_path, best_checkpoint_path)

    ev_step, true_step, perturbation_id, dev_error = train_loop(FLAGS, 
                            data_manager, perturbed_model, optimizer, 
                            trainer, training_data_iter, eval_iterators, 
                            logger, true_step, best_dev_error, perturbation_id, ev_step, header, root_id, vocabulary)

    logger.Log("Best dev accuracy of model: Step %i, %f" % (true_step, 1. - dev_error))

    queue.put((ev_step, true_step, perturbation_id, dev_error))


def perturb_model(model, random_seed):
    models = []
    np.random.seed(random_seed)
    for i in range(FLAGS.es_num_episodes):
        pert_model = model
        anti_model = model
        pert_model.load_state_dict(model.state_dict()) 
        anti_model.load_state_dict(model.state_dict())
        if FLAGS.mirror == True: 
            for (k, v), (anti_k, anti_v) in zip(pert_model.spinn.evolution_params(), anti_model.spinn.evolution_params()):
                epsilon = np.random.normal(0, 1, v.size())
                v += torch.from_numpy(FLAGS.es_sigma * epsilon).float()
                anti_v += torch.from_numpy(FLAGS.es_sigma * -epsilon).float()
            models.append(pert_model)
            models.append(anti_model)
        else:
            for (k, v) in pert_model.spinn.evolution_params():
                epsilon = np.random.normal(0, 1, v.size())
                v += torch.from_numpy(FLAGS.es_sigma * epsilon).float()
            models.append(pert_model)
    return models


def generate_seeds_and_models(trainer, model, root_id, base=False):
    """
    Restore best checkpoint of the model and get pertirb the model.
    """
    if not base:
        root_name = FLAGS.experiment_name + "_p" + str(root_id)
        root_path = os.path.join(FLAGS.ckpt_path, root_name + ".ckpt")
        ev_step, true_step, dev_error = trainer.load(root_path)
    else:
        true_step = 0
    np.random.seed()
    random_seed = np.random.randint(2**20)
    models = perturb_model(model, random_seed)
    return random_seed, models, true_step


def get_pert_names(best=False):
    if best:
        exp_names = glob.glob1(FLAGS.ckpt_path, FLAGS.experiment_name + "*.ckpt_best")
    else:
        exp_names = glob.glob1(FLAGS.ckpt_path, FLAGS.experiment_name + "*.ckpt")

    return exp_names


def restore(logger, trainer, queue, FLAGS, name, path):
    """
    Restore models
    """
    perturbation_id = name[-1]
    logger.Log("Restoring best checkpoint of perturbed model %s." % perturbation_id)
    ev_step, true_step, dev_error = trainer.load(path)
    queue.put((ev_step, true_step, perturbation_id, dev_error))


def run(only_forward=False):
    #logger = afs_safe_logger.ProtoLogger(log_path(FLAGS))
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
    num_classes = len(data_manager.LABEL_MAP)

    model, optimizer, trainer = init_model(
        FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager, header)

    # Checking if experiment with petrurbation id 0 has a checkpoint
    perturbation_name = FLAGS.experiment_name + "_p" + '0'
    best_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, perturbation_name, best=True)
    standard_checkpoint_path = get_checkpoint_path(FLAGS.ckpt_path, perturbation_name, best=False)

    ckpt_names = []
    if os.path.isfile(best_checkpoint_path):
        logger.Log("Found best checkpoints, they will be restored.")
        ckpt_names = get_pert_names(best=True)
    elif os.path.isfile(standard_checkpoint_path):
        logger.Log("Found standard checkpoints, they will be restored.")
        ckpt_names = get_pert_names(best=False)
    else:
        assert not only_forward, "Can't run an eval-only run without best checkpoints. Supply best checkpoint(s)."
        true_step = 0
        best_dev_error = 1.0
        reload_ev_step = 0
    
    if FLAGS.mirror: 
        true_num_episodes = FLAGS.es_num_episodes * 2
    else:
        true_num_episodes = FLAGS.es_num_episodes

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

    logger.LogHeader(header)  # Start log_entry logging.

    # Do an evaluation-only run.
    if only_forward:
        assert len(ckpt_names) != 0, "Can not run forward pass without best checkpoints supplied."
        log_entry = pb.SpinnEntry()

        restore_queue = mp.Queue()
        processes_restore = []
        while ckpt_names:
            pert_name = ckpt_names.pop()
            path = os.path.join(FLAGS.ckpt_path, pert_name)
            name = pert_name.replace('.ckpt_best', '')
            p_restore = mp.Process(target=restore, args=(logger, 
                                trainer, restore_queue,
                                FLAGS, name, path))
            p_restore.start()
            processes_restore.append(p_restore)
        assert len(ckpt_names) == 0
        
        results = [restore_queue.get() for p in processes_restore]
        assert results != 0 

        acc_order = [i[0] for i in sorted(enumerate(results), 
                                        key=lambda x:x[1][3])]
        best_id = acc_order[0]
        best_name = FLAGS.experiment_name + "_p" + str(best_id)
        print "Picking best perturbation/model %s to run evaluation" % (best_name)
        best_path = os.path.join(FLAGS.ckpt_path, best_name + ".ckpt_best")
        ev_step, true_step, dev_error = trainer.load(best_path)

        for index, eval_set in enumerate(eval_iterators):
            log_entry.Clear()
            evaluate(FLAGS, model, data_manager, eval_set,
                log_entry, true_step, vocabulary, show_sample=True,
                eval_index=index)
            print(log_entry)
            logger.LogEntry(log_entry)

    # Train the model.
    else:
        # Restore model, i.e. perturbation spawns, from best checkpoint, if it exists, or standard checkpoint.
        # Get dev-set accuracies so we can select which models to use for the next evolution step.
        if len(ckpt_names) != 0:
            logger.Log("Restoring models from best or standard checkpoints")
            processes_restore = []
            restore_queue = mp.Queue()
            while ckpt_names:
                pert_name = ckpt_names.pop()
                path = os.path.join(FLAGS.ckpt_path, pert_name)
                name = pert_name.replace('.ckpt_best', '')
                p_restore = mp.Process(target=restore,
                                args=(logger, trainer, restore_queue,
                                    FLAGS, name, path))
                p_restore.start()
                processes_restore.append(p_restore)
            assert len(ckpt_names) == 0
            results = [restore_queue.get() for p in processes_restore]
            reload_ev_step = results[0][0] + 1  # the next evolution step

        else:
            id_ = "B"
            chosen_models = [(reload_ev_step, true_step, id_, best_dev_error)]
            base = True  # This is the "base" model
            results = []

        for ev_step in range(reload_ev_step, FLAGS.es_steps):
            logger.Log("Evolution step: %i" % ev_step)

            # Downsample dev-set for evaluation runs during training
            eval_iterators_ = []
            if FLAGS.eval_sample_size != None:
                for file in eval_iterators:
                    eval_filename = eval_iterators[0][0]
                    eval_batches = eval_iterators[0][1]
                    full = len(eval_batches)
                    subsample = int(full * FLAGS.eval_sample_size)
                    eval_batches = random.sample(eval_batches, subsample)
                    eval_iterators_.append((eval_filename, eval_batches))
            else:
                eval_iterators_ = eval_iterators

            # Choose root models for next generation using dev-set accuracy
            if len(results) != 0:
                base = False
                chosen_models = []
                acc_order = [i[0] for i in sorted(enumerate(results),
                                            key=lambda x:x[1][3])]
                for i in range(FLAGS.es_num_roots):
                    id_ = acc_order[i]
                    logger.Log(
                        "Picking model %s to perturb for next evolution step." %
                        results[id_][2])
                    chosen_models.append(results[id_])

            # Flush results from previous generatrion
            results = []
            processes = []
            queue = mp.Queue()
            all_seeds, all_models, all_roots, all_steps, all_dev_errs = ([] for i in range(5))
            for chosen_model in chosen_models:
                perturbation_id = chosen_model[2]
                random_seed, models, true_step = generate_seeds_and_models(
                    trainer, model, perturbation_id, base=base)
                for i in range(len(models)):
                    all_seeds.append(random_seed)
                    all_steps.append(true_step) #chosen_model[1])
                    all_dev_errs.append(chosen_model[3])
                    all_roots.append(perturbation_id)
                all_models += models
            assert len(all_seeds) == len(all_models)
            assert len(all_steps) == len(all_seeds)

            perturbation_id = 0
            j = 0
            while all_models:
                perturbed_model = all_models.pop()
                true_step = all_steps.pop()
                best_dev_error = all_dev_errs.pop()
                root_id = all_roots.pop()
                p = mp.Process(target=rollout, args=(queue,
                                 perturbed_model, FLAGS, data_manager,
                                 model, optimizer, trainer, training_data_iter,
                                 eval_iterators_, logger, true_step,
                                 best_dev_error, perturbation_id, ev_step, header, root_id, vocabulary))
                p.start()
                processes.append(p)
                perturbation_id += 1
                j += 1
            assert len(all_models) == 0, "All models where not trained!"

            for p in processes:
                p.join()

            results = [queue.get() for p in processes]

            # Check to ensure the correct number of models where trained and saved
            if ev_step == 0:
                assert len(results) == true_num_episodes
            else:
                assert len(results) == true_num_episodes * FLAGS.es_num_roots


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS)

    if FLAGS.model_type == "RLSPINN":
        raise Exception(
            "Please use rl_classifier.py instead of supervised_classifier.py for RLSPINN.")

    if not FLAGS.evolution:
        raise Exception(
            "Please use supervised_classifier.py instead of es_classifier.py. This classifier trains the parser using evolution strategy.")

    run(only_forward=FLAGS.expanded_eval_only_mode)
