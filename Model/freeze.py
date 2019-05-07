from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pickle
import random
from tqdm import tqdm

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
from vgg16 import Vgg16

VGG_MEAN = [103.939, 116.779, 123.68]

FLAGS = None

def main(FLAGS):

    v = Vgg16()
    # var_list = [v for k, v in v.var_dict.items()]
    checkpoint = tf.train.Checkpoint(model=v)
    checkpoint.restore(FLAGS.start_checkpoint)
    call = v.__call__.get_concrete_function(tf.TensorSpec([None, 224, 224, 3], tf.float32))
    tf.saved_model.save(v, "/home/applevidur/tf_model/vgg16_with_signature/1", signatures=call)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_checkpoint",
        type=str,
        default=None,
        help="Start checkpoint")
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)