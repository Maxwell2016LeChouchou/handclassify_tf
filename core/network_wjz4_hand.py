
import tensorflow as tf
import tensorflow.contrib.slim as slim

from core.network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, dwconvb, is_trainable, dense
import core
import numpy as np

def build_network(input, trainable):
    is_trainable(trainable)

    # Use 144 x 144

    # 144x144
    net = convb(input, 3, 3, 8, 2, name="Conv2d_0")
    
   
    # 72
    branch_0 = slim.stack(net, inverted_bottleneck,
                              [
                                  (1    , 16, 0, 3),
                              ], scope="part0")

    # 36
    branch_1 = slim.stack(branch_0, inverted_bottleneck,
                              [
                                  (1    , 16, 1, 3),
                                  (1    , 16, 0, 3),
                              ], scope="part1")

    # 18
    branch_2 = slim.stack(branch_1, inverted_bottleneck,
                              [
                                  (1    , 32, 1, 3),
                                  (1    , 32, 0, 3),
                                  (1    , 32, 0, 3),
                                  (1    , 32, 0, 3),
                              ], scope="part2")

    # 9
    branch_3 = slim.stack(branch_2, inverted_bottleneck,
                              [
                                  (1    , 64, 1, 3),
                                  (1    , 64, 0, 3),
                                  (1    , 64, 0, 3),
                              ], scope="part3")


    # 7
    net = dwconvb(branch_3, 3, 3, 1, name="Conv2d_1_dw",padding="VALID")
    branch_4 = convb(net, 1, 1, 128, 1, name="Conv2d_1_sep")

    # 5
    net = dwconvb(branch_4, 3, 3, 1, name="Conv2d_2_dw",padding="VALID")
    branch_5 = convb(net, 1, 1, 256, 1, name="Conv2d_2_sep")

    # 3
    net = dwconvb(branch_5, 3, 3, 1, name="Conv2d_3_dw",padding="VALID")
    branch_6 = convb(net, 1, 1, 512, 1, name="Conv2d_3_sep")

    # 1
    net = dwconvb(branch_6, 3, 3, 1, name="Conv2d_4_dw",padding="VALID")
    branch_7 = convb(net, 1, 1, 512, 1, name="fc1", relu=True)

    avg_pool1 = slim.avg_pool2d(branch_6, [branch_6.get_shape()[1], branch_6.get_shape()[2]], stride=1)
    #print(avg_pool1.name, avg_pool1.get_shape())

    avg_pool2 = slim.avg_pool2d(branch_3, [branch_3.get_shape()[1], branch_3.get_shape()[2]], stride=1)
    #print(avg_pool2.name, avg_pool2.get_shape())

    s1 = branch_7
    s2 = avg_pool1
    s3 = avg_pool2
    multi_scale = tf.concat([s1, s2, s3], -1)#(?,1,1,576)
    #print(multi_scale.shape)

    hand = convb(multi_scale, 1, 1, 512, 1, name="hand_fc", relu=True)
    
    hand = slim.convolution2d(hand, 9, [1, 1], activation_fn=None, normalizer_fn=None, scope='hand_fc2')
    hand = tf.nn.softmax(hand,axis=3,name='hand_softmax')
    hand = slim.flatten(hand,scope='hand_out')

    return hand

