import torch
import torch.optim as optim
from spinn.util.blocks import the_gpu
from spinn.util.misc import recursively_set_device

import os

def get_checkpoint_path(FLAGS, suffix=".ckpt", best=False):
    # Set checkpoint path.

    if FLAGS.expanded_eval_only_mode and FLAGS.expanded_eval_only_mode_use_best_checkpoint:
        best = True

    if FLAGS.ckpt_path.endswith(".ckpt") or FLAGS.ckpt_path.endswith(".ckpt_best"):
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
    return checkpoint_path


class ModelTrainer(object):
    def __init__(self, model, logger, epoch_length, vocabulary, FLAGS):
        self.model = model
        self.logger = logger
        self.epoch_length = epoch_length
        self.vocabulary = vocabulary

        self.logger.Log('One epoch is ' + str(self.epoch_length) + ' steps.')

        self.dense_parameters = [param for name, param in model.named_parameters() if name not in ["embed.embed.weight"]]
        self.sparse_parameters = [param for name, param in model.named_parameters() if name in ["embed.embed.weight"]]
        self.optimizer_type = FLAGS.optimizer_type
        self.l2_lambda = FLAGS.l2_lambda
        self.ckpt_step = FLAGS.ckpt_step
        self.ckpt_on_best_dev_error = FLAGS.ckpt_on_best_dev_error
        self.learning_rate_decay_when_no_progress = FLAGS.learning_rate_decay_when_no_progress
        self.training_data_length = None
        self.eval_interval_steps = FLAGS.eval_interval_steps

        # GPU support.
        self.gpu = FLAGS.gpu
        the_gpu.gpu = FLAGS.gpu
        if self.gpu >= 0:
            model.cuda()
        else:
            model.cpu()

        self.step = 0
        self.best_dev_error = 1.0
        self.best_dev_step = 0

        self.optimizer_reset(FLAGS.learning_rate)

        self.standard_checkpoint_path = get_checkpoint_path(FLAGS)
        self.best_checkpoint_path = get_checkpoint_path(FLAGS, best=True)

        # Load checkpoint if available.
        if FLAGS.load_best and os.path.isfile(best_checkpoint_path):
            self.logger.Log("Found best checkpoint, restoring.")
            self.load(self.best_checkpoint_path, cpu=FLAGS.gpu < 0)
            self.logger.Log(
                "Resuming at step: {} with best dev accuracy: {}".format(
                    self.step, 1. - self.best_dev_error))
        elif os.path.isfile(self.standard_checkpoint_path):
            self.logger.Log("Found checkpoint, restoring.")
            self.load(self.standard_checkpoint_path, cpu=FLAGS.gpu < 0)
            self.logger.Log(
                "Resuming at step: {} with best dev accuracy: {}".format(
                    self.step, 1. - self.best_dev_error))


    def optimizer_reset(self, learning_rate):
        self.learning_rate = learning_rate

        if self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.dense_parameters, lr=learning_rate, 
                weight_decay=self.l2_lambda)

            if len(self.sparse_parameters) > 0:
                self.sparse_optimizer = optim.SparseAdam(self.sparse_parameters, lr=learning_rate)
            else:
                self.sparse_optimizer = None
        elif self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.dense_parameters, lr=learning_rate, 
                weight_decay=self.l2_lambda)
            if len(self.sparse_parameters) > 0:
                self.sparse_optimizer = optim.SGD(self.sparse_parameters, lr=learning_rate)
            else:
                self.sparse_optimizer = None

        if the_gpu() >= 0:
            recursively_set_device(self.optimizer.state_dict(), the_gpu())
            if self.sparse_optimizer is not None:
                recursively_set_device(self.sparse_optimizer.state_dict(), the_gpu())

    def optimizer_step(self):
        self.optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        self.step += 1

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

    def new_dev_accuracy(self, acc):
        # Track best dev error
        if (1 - acc) < 0.99 * self.best_dev_error:
            self.best_dev_error = 1 - acc
            self.best_dev_step = self.step
            if self.ckpt_on_best_dev_error and self.step > self.ckpt_step:
                self.logger.Log(
                    "Checkpointing with new best dev accuracy of %f" %
                    acc)
                self.save(self.best_checkpoint_path)

        # Learning rate decay
        if self.learning_rate_decay_when_no_progress != 1.0:
            last_epoch_start = self.step - (self.step % self.epoch_length)
            if self.step - last_epoch_start <= self.eval_interval_steps and self.best_dev_step < (last_epoch_start - self.epoch_length):
                    self.logger.Log('No improvement after one epoch. Lowering learning rate.')
                    self.optimizer_reset(self.learning_rate * self.learning_rate_decay_when_no_progress)

    def checkpoint(self):
        self.logger.Log("Checkpointing.")
        self.save(self.standard_checkpoint_path)

    def save(self, filename):
        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=-1)
            recursively_set_device(self.optimizer.state_dict(), gpu=-1)

        # Always sends Tensors to CPU.
        save_dict = {
            'step': self.step,
            'best_dev_error': self.best_dev_error,
            'best_dev_step': self.best_dev_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocabulary': self.vocabulary
            }
        if self.sparse_optimizer is not None:
            save_dict['sparse_optimizer_state_dict'] = self.sparse_optimizer.state_dict()
        torch.save(save_dict, filename)

        if the_gpu() >= 0:
            recursively_set_device(self.model.state_dict(), gpu=the_gpu())
            recursively_set_device(self.optimizer.state_dict(), gpu=the_gpu())

    def load(self, filename, cpu=False):
        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        # HACK: Compatability for saving supervised SPINN and loading RL SPINN.
        if 'baseline' in list(self.model.state_dict().keys(
        )) and 'baseline' not in model_state_dict:
            model_state_dict['baseline'] = torch.FloatTensor([0.0])

        if 'embed.embed.weight' in model_state_dict:
            loaded_embeddings = model_state_dict['embed.embed.weight']
            del(model_state_dict['embed.embed.weight'])

            count = 0
            for word in checkpoint['vocabulary']:
                if word in self.vocabulary:
                    self_index = self.vocabulary[word]
                    loaded_index = checkpoint['vocabulary'][word]
                    self.model.embed.embed.weight.data[self_index, :] = loaded_embeddings[loaded_index, :]
                    count += 1
            self.logger.Log('Restored ' + str(count) + ' words from checkpoint.')

        # "    (embed): Embedding(5759, 300, sparse=True)"

        self.model.load_state_dict(model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.gpu >= 0:
            for state in self.optimizer.state.values():
                for k, v in staste.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if self.sparse_optimizer is not None:
            self.sparse_optimizer.load_state_dict(checkpoint['sparse_optimizer_state_dict'])

        self.step = checkpoint['step']
        self.best_dev_step = checkpoint['best_dev_step']
        self.best_dev_error = checkpoint['best_dev_error']
