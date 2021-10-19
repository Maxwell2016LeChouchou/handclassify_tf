# -*- coding: utf-8 -*-

import tensorflow as tf
import os,sys
import platform
import time
import numpy as np
import argparse
sys.path.append(os.getcwd())

from datetime import datetime
from core.networks_hand import get_network
from core.dataset_hand import get_train_dataset_pipeline, get_valid_dataset_pipeline

def get_loss_and_output(net_name, batch_size, input_image, input_hand_label, reuse_variables=None):
    #num_keep_radio = 0.5

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        pred_hand = get_network(net_name, input_image, True)

    #hand
    hand_loss = tf.reduce_sum(tf.square(pred_hand-input_hand_label),axis=1)
    hand_loss = tf.reduce_sum(hand_loss) / tf.cast(batch_size,dtype=np.float32)

    
    hand_loss = hand_loss * 100
    total_loss = hand_loss
    return total_loss, hand_loss


def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(args):
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu_num = len(args.gpus.split(','))

    model_dir = args.model_dir
    log_dir = args.log_dir
    net_name = args.net_name
    batch_size = args.batch_size
    per_update_tensorboard_step = args.per_update_tensorboard_step
    per_saved_model_step = args.per_saved_model_step
    training_name = net_name
    if args.use_gray:
        training_name = training_name + '-gray'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    np.random.seed(args.seed)

		
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        # train_dataset = get_train_dataset_pipeline(args.image_path, args.train_file, args.image_size, args.image_size, 
        #                                         batch_size, args.max_epoch, 100, args.thread_num, args.use_gray)
        # valid_dataset = get_valid_dataset_pipeline(args.image_path, args.valid_file, args.image_size, args.image_size, 
        #                                         batch_size, args.max_epoch, 100, args.thread_num, args.use_gray)
        
        train_dataset = get_train_dataset_pipeline(args.train_file, args.image_size, args.image_size, 
                                                batch_size, args.max_epoch, 100, args.thread_num, args.use_gray)
        valid_dataset = get_valid_dataset_pipeline(args.valid_file, args.image_size, args.image_size, 
                                                batch_size, args.max_epoch, 100, args.thread_num, args.use_gray)
                
        train_iterator = train_dataset.make_one_shot_iterator()
        valid_iterator = valid_dataset.make_one_shot_iterator()

        
        handle = tf.placeholder(tf.string, shape=[])
        input_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(args.lr, global_step,
                                                   decay_steps=10000, decay_rate=args.decay_rate, staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        #opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        reuse_variable = False

        if platform.system() == 'Darwin':
            # cpu (mac only)
            with tf.device("/cpu:0"):
                with tf.name_scope("CPU_0"):
                    input_image, input_hand_label = input_iterator.get_next()

                    final_loss, hand_loss = get_loss_and_output(net_name, batch_size, input_image, input_hand_label, reuse_variable)
                    reuse_variable = True
                    grads = opt.compute_gradients(final_loss)
                    tower_grads.append(grads)
        else:
            # multiple gpus
            for i in range(gpu_num):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("GPU_%d" % i):
                        input_image, input_hand_label = input_iterator.get_next()
                        #print(input_heat.shape)
                        final_loss, hand_loss = get_loss_and_output(net_name, batch_size, input_image, input_hand_label, reuse_variable)
                        reuse_variable = True
                        grads = opt.compute_gradients(final_loss)
                        tower_grads.append(grads)

		
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)


        
        saver = tf.train.Saver(max_to_keep=100)

        tf.summary.scalar("final_loss", final_loss)
        tf.summary.scalar("hand_loss", hand_loss)
        summary_merge_op = tf.summary.merge_all()

        if args.resume == 0:
            init = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if args.resume == 0:
                init.run()
            else:
                #saver=tf.train.import_meta_graph(os.path.join(model_path, training_name, 'model-%d.meta'%resume))
                saver.restore(sess,tf.train.latest_checkpoint(os.path.join(model_dir, training_name)))
            train_handle = sess.run(train_iterator.string_handle())
            valid_handle = sess.run(valid_iterator.string_handle())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            summary_writer_train = tf.summary.FileWriter(os.path.join(log_dir, training_name, 'train'), sess.graph)
            summary_writer_valid = tf.summary.FileWriter(os.path.join(log_dir, training_name, 'valid'))
            total_step_num = args.num_train_samples * args.max_epoch // (batch_size * gpu_num)
            print("Start training...")
            for step in range(total_step_num):
                start_time = time.time()
                _, final_loss_val, hand_loss_val = sess.run(
                                                  [train_op, final_loss, hand_loss],
                                                  feed_dict={handle: train_handle}
                )
                duration = time.time() - start_time

                if step != 0 and step % per_update_tensorboard_step == 0:
                    summary_final_loss_val = tf.Summary(value=[tf.Summary.Value(tag="final_loss", simple_value=final_loss_val)])
                    summary_hand_loss_val = tf.Summary(value=[tf.Summary.Value(tag="hand_loss", simple_value=hand_loss_val)])
                    summary_writer_train.add_summary(summary_final_loss_val, step)
                    summary_writer_train.add_summary(summary_hand_loss_val, step)
					

                    # print train info
                    num_examples_per_step = batch_size * gpu_num
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / gpu_num
                    format_str = ('%s: step %d, final=%.5f, hand=%.5f (%.1f examples/s; %.3f s/batch)')
                    print(format_str % (datetime.now(), step, final_loss_val, hand_loss_val, examples_per_sec, sec_per_batch))

                    # tensorboard visualization
                    final_valid_loss, hand_valid_loss = sess.run(
                                                   [final_loss, hand_loss], feed_dict={handle: valid_handle})
                    summary_final_loss_val = tf.Summary(value=[tf.Summary.Value(tag="final_loss", simple_value=final_valid_loss)])
                    summary_hand_loss_val = tf.Summary(value=[tf.Summary.Value(tag="hand_loss", simple_value=hand_valid_loss)])
                    summary_writer_valid.add_summary(summary_final_loss_val, step)
                    summary_writer_valid.add_summary(summary_hand_loss_val, step)

                # save model
                if step != 0 and step % per_saved_model_step == 0:
                    checkpoint_path = os.path.join(model_dir, training_name, 'model')
                    saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_name', dest='net_name',type=str, default='zq_hand_1')
    # parser.add_argument('--image_path', dest='image_path',type=str, default='./')
    parser.add_argument('--train_file', dest='train_file',type=str, default='prepare_data/96/hand.txt')
    parser.add_argument('--valid_file', dest='valid_file',type=str, default='prepare_data/96/hand.txt')
    parser.add_argument('--gpus', dest='gpus',type=str, default='7')
    parser.add_argument('--thread_num', dest='thread_num',type=int, default=10)
    parser.add_argument('--seed',dest='seed',type=int, default=666)
    parser.add_argument('--max_epoch', dest='max_epoch',type=int, default=1000)
    parser.add_argument('--num_train_samples', dest='num_train_samples',type=int, default=680000)
    parser.add_argument('--image_size', dest='image_size',type=int, default=96)
    parser.add_argument('--use_gray', dest='use_gray',type=bool, default=False)
    parser.add_argument('--batch_size', dest='batch_size',type=int, default=512)
    parser.add_argument('--model_dir', dest='model_dir',type=str, default='models')
    parser.add_argument('--log_dir', dest='log_dir',type=str, default='models')
    parser.add_argument('--lr', dest='lr',type=float, default=0.01)
    parser.add_argument('--decay_rate', dest='decay_rate',type=float, default=0.95)
    parser.add_argument('--per_saved_model_step', dest='per_saved_model_step',type=int, default=5000)
    parser.add_argument('--per_update_tensorboard_step', dest='per_update_tensorboard_step',type=int, default=20)
    parser.add_argument('--resume', dest='resume', type=int, default=0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
