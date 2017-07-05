import unittest
import numpy as np
import tempfile

from spinn.pyramid import Pyramid

import spinn.pyramid
from spinn.util.blocks import ModelTrainer

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from spinn.util.test import MockModel, default_args, get_batch, compare_models


class SPINNTestCase(unittest.TestCase):

    def test_save_load_model(self):
        model_to_save = MockModel(Pyramid, default_args())
        model_to_load = MockModel(Pyramid, default_args())

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        torch.save(model_to_save.state_dict(), temp.name)
        model_to_load.load_state_dict(torch.load(temp.name))

        compare_models(model_to_save, model_to_load)

        # Cleanup temporary file.
        temp.close()

    def test_save_sup_load_rl(self):
        pass

        model_to_save = MockModel(spinn.pyramid.Pyramid, default_args())
        opt_to_save = optim.SGD(model_to_save.parameters(), lr=0.1)
        trainer_to_save = ModelTrainer(model_to_save, opt_to_save)

        model_to_load = MockModel(spinn.pyramid.Pyramid, default_args())
        opt_to_load = optim.SGD(model_to_load.parameters(), lr=0.1)
        trainer_to_load = ModelTrainer(model_to_load, opt_to_load)

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        trainer_to_save.save(temp.name, 0, 0)
        trainer_to_load.load(temp.name)

        compare_models(model_to_save, model_to_load)

        # Cleanup temporary file.
        temp.close()

    def test_init_models(self):
        MockModel(spinn.pyramid.Pyramid, default_args())

        MockModel(spinn.pyramid.Pyramid, default_args(use_sentence_pair=True))

    def test_training(self):
        model = MockModel(Pyramid, default_args())

        X, transitions = get_batch()

        class Projection(nn.Module):
            def forward(self, x):
                return x[:, :default_args()['model_dim']]

        class mlp(nn.Module):
            def forward(self, x):
                return x

        def selection_fn(selection_input):
            return Variable(torch.rand(selection_input.data.size()[0], 1))  # Compose in random order

        class Reduce(nn.Module):
            def forward(self, lefts, rights):
                return lefts + rights

        model.encode = Projection()
        model.composition_fn = Reduce()
        model.selection_fn = selection_fn
        model.mlp = mlp()

        outputs = model(X, transitions, pyramid_temperature_multiplier=0.0000001)

        assert outputs[0][0].data[0] == (3 + 1 + 2 + 1)
        assert outputs[1][0].data[0] == (3 + 2 + 4 + 5)


    def test_soft_eval(self):
        args = default_args()
        args['test_temperature_multiplier'] = 0.0000001
        model = MockModel(Pyramid, args)

        X, transitions = get_batch()

        class Projection(nn.Module):
            def forward(self, x):
                return x[:, :default_args()['model_dim']]

        class mlp(nn.Module):
            def forward(self, x):
                return x

        def selection_fn(selection_input):
            return Variable(torch.rand(selection_input.data.size()[0], 1))  # Compose in random order

        class Reduce(nn.Module):
            def forward(self, lefts, rights):
                return lefts + rights

        model.encode = Projection()
        model.composition_fn = Reduce()
        model.selection_fn = selection_fn
        model.mlp = mlp()

        model.eval()
        outputs = model(X, transitions)

        assert outputs[0][0].data[0] == (3 + 1 + 2 + 1)
        assert outputs[1][0].data[0] == (3 + 2 + 4 + 5)


    def test_hard_eval(self):
        args = default_args()
        args['test_temperature_multiplier'] = 0.0
        model = MockModel(Pyramid, args)

        X, transitions = get_batch()

        class Projection(nn.Module):
            def forward(self, x):
                return x[:, :default_args()['model_dim']]

        class mlp(nn.Module):
            def forward(self, x):
                return x

        def selection_fn(selection_input):
            return Variable(torch.rand(selection_input.data.size()[0], 1))  # Compose in random order

        class Reduce(nn.Module):
            def forward(self, lefts, rights):
                return lefts + rights

        model.encode = Projection()
        model.composition_fn = Reduce()
        model.selection_fn = selection_fn
        model.mlp = mlp()

        model.eval()
        outputs = model(X, transitions)

        assert outputs[0][0].data[0] == (3 + 1 + 2 + 1)
        assert outputs[1][0].data[0] == (3 + 2 + 4 + 5)

if __name__ == '__main__':
    unittest.main()
