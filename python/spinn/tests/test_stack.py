import unittest
import numpy as np
import tempfile
import gflags
import sys

from spinn.spinn_core_model import BaseModel
from spinn.models.base import get_flags

import spinn.spinn_core_model
import spinn.rl_spinn
from spinn.util.trainer import ModelTrainer

# PyTorch
import torch
import torch.nn as nn

from spinn.util.test import MockModel, default_args, get_batch, compare_models
from spinn.util import afs_safe_logger

FLAGS = gflags.FLAGS

class SPINNTestCase(unittest.TestCase):

    def test_save_load_model(self):
        model_to_save = MockModel(BaseModel, default_args())
        model_to_load = MockModel(BaseModel, default_args())

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        torch.save(model_to_save.state_dict(), temp.name)
        model_to_load.load_state_dict(torch.load(temp.name))

        compare_models(model_to_save, model_to_load)

        # Cleanup temporary file.
        temp.close()

    def test_save_sup_load_rl(self):
        pass

        model_to_save = MockModel(
            spinn.spinn_core_model.BaseModel,
            default_args())

        # Parse command line flags.
        get_flags()
        FLAGS(sys.argv)

        log_temp = tempfile.NamedTemporaryFile()
        ckpt_temp = tempfile.NamedTemporaryFile()

        logger = afs_safe_logger.ProtoLogger(log_temp.name)
        FLAGS.ckpt_path = '.'

        trainer_to_save = ModelTrainer(model_to_save, logger, FLAGS)

        model_to_load = MockModel(spinn.rl_spinn.BaseModel, default_args())
        trainer_to_load = ModelTrainer(model_to_load, logger, FLAGS)

        # Save to and load from temporary file.
        trainer_to_save.save(ckpt_temp.name)
        trainer_to_load.load(ckpt_temp.name, cpu=True)

        compare_models(model_to_save, model_to_load)

        # Cleanup temporary file.
        ckpt_temp.close()
        log_temp.close()

    def test_init_models(self):
        MockModel(spinn.spinn_core_model.BaseModel, default_args())
        MockModel(spinn.rl_spinn.BaseModel, default_args())

        MockModel(
            spinn.spinn_core_model.BaseModel,
            default_args(
                use_sentence_pair=True))
        MockModel(
            spinn.rl_spinn.BaseModel,
            default_args(
                use_sentence_pair=True))

    def test_basic_stack(self):
        model = MockModel(BaseModel, default_args())

        X, transitions = get_batch()

        class Projection(nn.Module):
            def forward(self, x):
                return x[:, :default_args()['model_dim']]

        class Reduce(nn.Module):
            def forward(self, lefts, rights, tracking):
                batch_size = len(lefts)
                return torch.chunk(
                    torch.cat(
                        lefts,
                        0) -
                    torch.cat(
                        rights,
                        0),
                    batch_size,
                    0)

        model.encode = Projection()
        model.spinn.reduce = Reduce()

        model(X, transitions)
        outputs = model.spinn_outp[0]

        assert outputs[0][0].data[0] == (3 - (1 - (2 - 1)))
        assert outputs[1][0].data[0] == ((3 - 2) - (4 - 5))

    def test_validate_transitions_cantskip(self):
        model = MockModel(BaseModel, default_args())

        # To Test:
        # 1. Cant SKIP
        # 2. Cant SHIFT
        # 3. Cant REDUCE
        # 4. No change SHIFT
        # 5. No change REDUCE

        bufs = [
            [None],
            [],
            [None],
            [None],
            [None],
            [],
        ]

        stacks = [
            [None],
            [None],
            [None],
            [],
            [],
            [None, None],
        ]

        transitions = [
            2, 1, 0, 0, 0, 1
        ]
        preds = np.array([
            0, 0, 1, 1, 0, 1
        ]).astype(np.int32)

        ret, _ = model.spinn.validate(
            transitions, preds, stacks, bufs, zero_padded=False)
        expected = np.array([
            2, 1, 0, 0, 0, 1
        ], dtype=np.int32)

        assert all(
            p == e for p, e in zip(
                ret.ravel().tolist(), expected.ravel().tolist())), "gold: {}\npreds: {}".format(
            expected, ret)


if __name__ == '__main__':
    unittest.main()
