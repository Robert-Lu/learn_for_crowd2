#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

import pdb
from lc import *
from tensorflow.contrib.layers import *
from tensorflow.contrib.keras.python.keras.layers import *

config.LEARNING_RATE = 0.01
config.DECAY_STEP = 75
config.DECAY_RATE = 0.975
config.L2_LAMBDA = 0.05
config.STOP_THRESHOLD = -1
config.RESTORE_FROM = "08-10-17_04_10"

d = {"name": "lambda_1", "discription": "TEST_D", }
l = Loader(d)

'''
  def apply(self, inputs, *args, **kwargs):
    Apply the layer on a input.

    This simply wraps `self.__call__`.

    Arguments:
      inputs: Input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
'''


def max_out(inputs, num_units, axis=None):

    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def five_layers_lrelu(x, ref_y, test):
    test = None if not test else True
    # lrelu = LeakyReLU()
    hid1 = fully_connected(
        x, 1000, activation_fn=lambda input: max_out(input, 500), reuse=test, scope="layer1")
    hid2 = fully_connected(
        hid1, 1000, activation_fn=lambda input: max_out(input, 250), reuse=test, scope="layer2")
    hid3 = fully_connected(
        hid2, 1000, activation_fn=lambda input: max_out(input, 125), reuse=test, scope="layer3")
    hid4 = fully_connected(
        hid3, 1000, activation_fn=lambda input: max_out(input, 50), reuse=test, scope="layer4")
    hid5 = fully_connected(
        hid4, 1000, activation_fn=lambda input: max_out(input, 25), reuse=test, scope="layer5")
    y = fully_connected(hid5, 1, activation_fn=tf.identity,
                        reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        analysis.add_L2_loss()
    else:
        analysis.add_RMSE_loss(y, ref_y, "test")

def linear(x, ref_y, test):
    test = None if not test else True
    y = fully_connected(x, 1, activation_fn=tf.identity,
                        reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        # analysis.add_L2_loss()
    else:

        analysis.add_RMSE_loss(y, ref_y, "test")


def apply_graph(graph):
    g1 = tf.Graph()
    with g1.as_default():
        x1, y1 = l.train()
        graph(x1, y1, False)

        x2, y2 = l.validation()
        graph(x2, y2, True)

        summarize_collection("trainable_variables")
        summarize_collection("losses")
    return g1


with apply_graph(five_layers_lrelu).as_default():
    train.simple_train(5000)
