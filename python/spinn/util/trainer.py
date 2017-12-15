import torch.optim as optim
from spinn.util.blocks import the_gpu
from spinn.util.misc import recursively_set_device


class ModelTrainer(object):
    def __init__(self, model, optimizer_type, learning_rate, l2_lambda, gpu):
        self.model = model
        self.dense_parameters = [param for name, param in model.named_parameters() if name not in ["embed.embed.weight"]]
        self.sparse_parameters = [param for name, param in model.named_parameters() if name in ["embed.embed.weight"]]
        self.optimizer_type = optimizer_type
        self.l2_lambda = l2_lambda

        # GPU support.
        self.gpu = gpu
        the_gpu.gpu = gpu
        if gpu >= 0:
            model.cuda()
        else:
            model.cpu()

        self.optimizer_reset(learning_rate)

        self.step = 0
        self.best_dev_error = 1.0
        self.best_dev_step = 0

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
            'optimizer_state_dict': self.optimizer.state_dict()
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
        if 'baseline' in self.model.state_dict().keys(
        ) and 'baseline' not in model_state_dict:
            model_state_dict['baseline'] = torch.FloatTensor([0.0])

        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.load_state_dict(checkpoint['sparse_optimizer_state_dict'])

        self.step = checkpoint['step']
        self.best_dev_step = checkpoint['best_dev_step']
        self.best_dev_error = checkpoint['best_dev_error']
