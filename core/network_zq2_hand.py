# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

from core.network_base_zq import max_pool, upsample, inverted_bottleneck, separable_conv, convb, dwconvb, is_trainable, dense
import core
import numpy as np

def build_network(input, trainable):
    is_trainable(trainable)

    # 96
    net = convb(input, 3, 3, 16, 2, name="Conv2d_0")
    
   
    # 48
    branch_0 = slim.stack(net, inverted_bottleneck,
                              [
                                  (1    , 32, 0, 3),
                              ], scope="part0")

    # 24
    branch_1 = slim.stack(branch_0, inverted_bottleneck,
                              [
                                  (1    , 32, 1, 3),
                                  (1    , 32, 0, 3),
                              ], scope="part1")

    # 12
    branch_2 = slim.stack(branch_1, inverted_bottleneck,
                              [
                                  (1    , 64, 1, 3),
                                  (1    , 64, 0, 3),
                                  (1    , 64, 0, 3),
                                  (1    , 64, 0, 3),
                              ], scope="part2")

    # 6
    branch_3 = slim.stack(branch_2, inverted_bottleneck,
                              [
                                  (1    , 128, 1, 3),
                                  (1    , 128, 0, 3),
                                  (1    , 128, 0, 3),
                              ], scope="part3")

    # 3
    branch_4 = slim.stack(branch_3, inverted_bottleneck,
                              [
                                  (1    , 256, 1, 3),
                                  (1    , 256, 0, 3),
                              ], scope="part4")

    # 1
    net = dwconvb(branch_4, 3, 3, 1, name="Conv2d_3_dw",padding="VALID")
    # net = convb(net, 1, 1, 256, 1, name="fc1", relu=True)

    # hand = convb(net, 1, 1, 256, 1, name="hand_fc", relu=True)
    
    net = convb(net, 1, 1, 512, 1, name = "fc1", relu=True)

    hand = convb(net, 1, 1, 512, 1, name = "hand_fc", relu=True)
    
    hand = slim.convolution2d(hand, 12, [1, 1], activation_fn=None, normalizer_fn=None, scope='hand_fc2')
    hand = tf.nn.softmax(hand,axis=3,name='hand_softmax')
    hand = slim.flatten(hand,scope='hand_out')

    return hand

