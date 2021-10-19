import tensorflow as tf 
import tensorflow.contrib.slim as slim 


def is_trainable(trainable=True):
    global _trainable
    _trainable = trainable

def separable_conv(self, input, k_h, k_w, c_o, stride, name, relu=True):
    with slim.arg_scope([slim.batch_norm], fused=common.batchnorm_fused):
        output = slim.separable_convolution2d(input,
                                              num_outputs = None,
                                              stride=stride,
                                              trainable=self.trainable,
                                              depth_multiplier=1.0,
                                              kernel_size=[k_h, k_w],
                                              activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(),
                                              # weight_iniializer=tf.truncated_normal_initializer(stddev=0.01),
                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.00004),
                                              biases_initializer=None,
                                              padding=DEFAULT_PADDING,
                                              scope='depthwise_' + name)
        output = slim.convolution2d(output,
                                    c_o,
                                    stride=1,
                                    kernel_size=[1, 1],
                                    activation_fn=tf.nn.relu if relu else None,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    biases_initializer=slim.init_ops.zeros_initializer(),
                                    normalizer_fn=slim.batch_norm,
                                    trainable=_trainable,
                                    weights_regularizer=tf.contrib.layers.l2_regularizer(common.regularizer_dsconv),
                                    # weights_regularizer=None,
                                    scope='pointwise_' + name)
        return output

