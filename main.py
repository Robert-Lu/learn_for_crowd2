#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

from lc import *
from tensorflow.contrib.layers import *
from tensorflow.contrib.keras.python.keras.layers import *

config.LEARNING_RATE = 0.001
config.DECAY_STEP = 50
config.DECAY_RATE = 0.90
config.L2_LAMBDA = 0.05
config.STOP_THRESHOLD = -1

d = {"name": "TEST_", "discription": "TEST_D"}
l = Loader(d)


def five_layers_lrelu(x, ref_y, test):
    test = None if not test else True
    lrelu = LeakyReLU()
    hid1 = fully_connected(
        x, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer1")
    hid2 = fully_connected(
        hid1, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer2")
    hid3 = fully_connected(
        hid2, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer3")
    hid4 = fully_connected(
        hid3, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer4")
    hid5 = fully_connected(
        hid4, 1000, activation_fn=lrelu.apply, reuse=test, scope="layer5")
    y = fully_connected(hid5, 1, activation_fn=tf.identity,
                        reuse=test, scope="fc")
    if not test:
        analysis.add_RMSE_loss(y, ref_y, "train")
        # analysis.add_L2_loss()
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
    train.simple_train(50000)