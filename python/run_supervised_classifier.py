
import sys

import gflags

from spinn.models import supervised_classifier

from spinn.models.base import get_flags
from spinn.models.base import flag_defaults

FLAGS = gflags.FLAGS


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS)

    if FLAGS.model_type == "RLSPINN":
        raise Exception(
            "Please use rl_classifier.py instead of supervised_classifier.py for RLSPINN.")

    supervised_classifier.run(only_forward=FLAGS.expanded_eval_only_mode)
