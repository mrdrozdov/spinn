import unittest
import tempfile
import math

from nose.plugins.attrib import attr
import numpy as np

import pytest

from spinn import util
from spinn.fat_stack import SPINN

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import spinn.util.blocks as blocks

from spinn.util.blocks import DefaultUniformInitializer as SimpleInitializer
from spinn.util.blocks import ZeroInitializer as SimpleBiasInitializer
from spinn.util.blocks import HeKaimingLinear as CustomLinear
from spinn.util.blocks import Linear

from spinn.util.test import compare_models


class MockModel(nn.Module):
    def __init__(self, scalar=11):
        super(MockModel, self).__init__()
        self.layer = nn.Linear(2, 2)
        self.register_buffer('scalar', torch.Tensor([scalar]))


class PytorchTestCase(unittest.TestCase):

    def test_save_load_model(self):
        scalar = 11
        other_scalar = 0
        model_to_save = MockModel(scalar)
        model_to_load = MockModel(other_scalar)

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        torch.save(model_to_save.state_dict(), temp.name)
        model_to_load.load_state_dict(torch.load(temp.name))

        compare_models(model_to_save, model_to_load)

        # Check value of scalars.
        assert model_to_save.scalar[0] == 11
        assert model_to_save.scalar[0] == model_to_load.scalar[0]
        
        # Cleanup temporary file.
        temp.close()


    def test_custom_init(self):

        # Concrete class that uses custom init.
        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.l = CustomLinear(10, 10)

        model_to_save = MyModel()
        model_to_load = MyModel()

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        torch.save(model_to_save.state_dict(), temp.name)
        model_to_load.load_state_dict(torch.load(temp.name))

        compare_models(model_to_save, model_to_load)
        
        # Cleanup temporary file.
        temp.close()

    def test_custom_functional_init(self):

        # Concrete class that uses custom init.
        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.l = Linear(SimpleInitializer, SimpleBiasInitializer)(10, 10)

        model_to_save = MyModel()
        model_to_load = MyModel()

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        torch.save(model_to_save.state_dict(), temp.name)
        model_to_load.load_state_dict(torch.load(temp.name))

        compare_models(model_to_save, model_to_load)
        
        # Cleanup temporary file.
        temp.close()


if __name__ == '__main__':
    unittest.main()
