# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
import os

def run_with_frozen_pb(img_path, input_w, input_h, use_gray, frozen_graph, output_node_names):

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with tf.gfile.GFile(frozen_graph, "rb") as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=""
    )

    graph = tf.get_default_graph()
    image = graph.get_tensor_by_name("image:0")
    output_node_names = output_node_names.split(',')
    output = list()
    for i in range(len(output_node_names)):
        output.append(graph.get_tensor_by_name("%s:0"%output_node_names[i]))
	
    show_width = 600
    show_height = 600
    if use_gray:
        image_0 = cv2.imread(img_path,0)
        h, w = image_0.shape
    else:
        image_0 = cv2.imread(img_path,1)
        if image_0.ndim == 2:
            image_0 = cv2.cvtColor(image_0,cv2.COLOR_GRAY2BGR)
        h, w, _ = image_0.shape
    if w != input_w or h != input_h:
        image_ = cv2.resize(image_0, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_ = image_0
    show_image = cv2.resize(image_0, (show_width, show_height))
    if use_gray:
        show_image = cv2.cvtColor(show_image, cv2.COLOR_GRAY2BGR)
    with tf.Session() as sess:
        tensor = image_.astype(np.float32)
        #tensor = (tensor - 127.5)*0.0078125
        if use_gray:
            tensor = tensor[:,:,np.newaxis]
        hand = sess.run(output, feed_dict={image: [tensor]})
        print(hand[0])
        
        cv2.imwrite("show.jpg",show_image)

        

if __name__ == '__main__':
    run_with_frozen_pb(
        "./prepare_data/96/hand/0_0.jpg",
        96,96,
        True,
        "./model-zq1-gray-40000.pb",
        'hand_out/flatten/Reshape'
    )
    #display_image()

