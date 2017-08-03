#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <stakeholder> All Rights Reserved
#
#
# File: /Users/hain/tmp/foo.py
# Author: Hai Liang Wang
# Date: 2017-08-01:15:03:17
#
#===============================================================================

'''
    maximum_likelihood_linear_regression_tensorflow.py
    https://stackoverflow.com/questions/41885665/maximum-likelihood-linear-regression-tensorflow
'''

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

graph = tf.Graph()

with graph.as_default():
    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        X = tf.placeholder("float", None)
        Y = tf.placeholder("float", None)
        theta_0 = tf.Variable(np.random.randn())
        theta_1 = tf.Variable(np.random.randn())
        var = tf.Variable(0.5)

        # y = theta_0 + (x * theta_1)
        hypothesis = tf.add(theta_0, tf.mul(X, theta_1))
        lhf = 1 * (50 * np.log(2*np.pi) + 50 * tf.log(var) + (1/(2*var)) * tf.reduce_sum(tf.pow(hypothesis - Y, 2)))
        op = tf.train.GradientDescentOptimizer(0.01).minimize(lhf)

        # Add variable initializer.
        init = tf.global_variables_initializer()

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    train_X = np.random.rand(100, 1) # all values [0-1)
    train_Y = train_X
    feed_dict = {X: train_X, Y: train_Y}
    num_steps = 100001

    for steps in range(num_steps):
        _, loss, v0, v1 = session.run([op, lhf, theta_0, theta_1], feed_dict=feed_dict)
        # since x == y, theta_0 should be 0 and theta_1 should be 1.
        print('Run step %s, loss %s, theta_0 %s, theta_1 %s' % (steps, loss, v0, v1))