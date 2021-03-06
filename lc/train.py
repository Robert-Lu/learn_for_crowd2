#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import tensorflow as tf
import time
import sys

from . import analysis
from . import config
from contextlib import contextmanager
from xilio import dump, write, append


tools = type("Tools", (), {})()

__all__ = ["simple_train", "training"]


def epoch_train(tools):
    """
    Do epoch train for one times.

    input: tools

    """
    sess = tools.sess
    optimizer = tools.optimizer

    infos, summary, g, _ = sess.run(tools.infos)
    print(config.INFOMESSAGE(infos))
    sys.stdout.flush()
    tools.reporter(summary, g)

    try:
        while True:
            sess.run(optimizer)
    except tf.errors.OutOfRangeError:
        pass
    return infos


@contextmanager
def training(merge_key=tf.GraphKeys.SUMMARIES, restore_from=None):
    with tf.Session() as sess:
        graph = tf.get_default_graph()

        path = config.DATANAME + "/" + time.strftime("%m-%d-%y_%H_%M")
        # path = config.DATANAME
        g = tf.Variable(0, name="global_step", trainable=False)
        with tf.name_scope("epoch_step"):
            e = tf.Variable(0, name="epoch_step", trainable=False)
            e_add = tf.assign(e, e + 1)

        fin_loss = analysis.fin_loss()
        with tf.name_scope("train"):
            learning_rate = tf.train.exponential_decay(
                float(config.LEARNING_RATE), e,
                float(config.DECAY_STEP), float(config.DECAY_RATE)
            )
            optimizer = (tf.train.AdamOptimizer(learning_rate)
                         .minimize(fin_loss, global_step=g))
        accur = graph.get_tensor_by_name("analysis/accuracy_train:0")
        val_accur = graph.get_tensor_by_name("analysis/accuracy_test:0")
        infos = [e, fin_loss, accur, val_accur]
        updates = [e_add, optimizer]

        writer = tf.summary.FileWriter(path + "/summary", graph)
        summary = tf.summary.merge_all(merge_key)
        saver = tf.train.Saver(tf.get_collection("trainable_variables"))
        print("check")
        if restore_from:
            print(config.DATANAME + "/" + restore_from)
            ckpt = tf.train.latest_checkpoint(
                config.DATANAME + "/" + restore_from)
            if ckpt:
                print("RESTROE")
                saver.restore(sess, ckpt)

        tools.path = path
        tools.sess = sess
        tools.graph = graph
        tools.saver = saver
        tools.infos = [infos, summary, g, updates]
        tools.optimizer = optimizer
        import types

        def reporter(self, summary, e):
            writer.add_summary(summary, e)
            writer.flush()
            saver.save(sess, path + "/chkpnt", g)

        tools.reporter = types.MethodType(reporter, tools)

        tf.global_variables_initializer().run(None, sess)
        tf.local_variables_initializer().run(None, sess)
        graph.finalize()
        yield tools


def dump_info(path, info):
    f = open(path)
    f.write(info)
    f.close()


def simple_train(epoch_steps):
    infos = []
    start_time = time.time()
    with training(restore_from=config.RESTORE_FROM) as tools:
        write(tools.path + "/description", config.DISCRIPTION + "\n")
        for i in range(epoch_steps):
            batch_init = tf.get_collection("batch_init")
            tools.sess.run(batch_init)
            infos.append(epoch_train(tools))
            if i > 5:
                recent = [x[1] for x in infos[-5:]]
                if np.std(recent) < config.STOP_THRESHOLD:
                    break
        dump(tools.path + "/trace", infos)
        duration = time.time() - start_time
        append(tools.path + "/description",
               "Time usage: " + time.strftime(
                   "%M minutes, %S seconds",
                   time.gmtime(duration)) + "\n")
    return infos
