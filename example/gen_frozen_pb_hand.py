# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os,sys
sys.path.append(os.getcwd())
from core.networks_hand import get_network



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--net_name', type=str, default='hand_net1', help='net model name')
parser.add_argument('--size_h', type=int, default=96)
parser.add_argument('--size_w', type=int, default=96)
parser.add_argument('--use_gray', type=bool, default=True)
parser.add_argument('--checkpoint', type=str, default='models/hand_net1/model-200000', help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='hand_out/flatten/Reshape')
parser.add_argument('--output_graph', type=str, default='./model-net1_hand.pb', help='output_freeze_path')

args = parser.parse_args()

if args.use_gray:
    input_node = tf.placeholder(tf.float32, shape=[1, args.size_h, args.size_w, 1], name="image")
else:
    input_node = tf.placeholder(tf.float32, shape=[1, args.size_h, args.size_w, 3], name="image")

with tf.Session() as sess:
    net = get_network(args.net_name, input_node, trainable=False)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    input_graph_def = tf.get_default_graph().as_graph_def()
    print(input_graph_def)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
