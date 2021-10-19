import tensorflow as tf
import numpy as np
import cv2
import os,sys
import numpy.random as npr
import multiprocessing
import random
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

train_file_list = []
train_hand_labels = []
valid_file_list = []
valid_hand_labels = []
im_width = 96
im_height = 96
im_channel = 3
class_num = 12

# def gen_data(image_path, anno_file):
def gen_data(anno_file):
    f = open(anno_file,'r')
    lines = f.readlines()
    f.close()
    #lines = lines[0:1000]
    random.shuffle(lines)
    all_filenames = []
    all_hand_labels = []
    imgids = []
    cur_id = 0
    for line in lines:
        #print(line)
        splits = line.strip().split()
        # path = os.path.join(image_path, splits[0])
        path = splits[0]
        hand = int(splits[1])
        hand_label = np.zeros(class_num,dtype=np.float32)
        hand_label[hand] = 1
        all_filenames.append(path)
        all_hand_labels.append(hand_label)
        imgids.append(cur_id)
        cur_id += 1
    return all_filenames, all_hand_labels, imgids

def _parse_data(imgid, is_train, im_width, im_height, use_gray):
    if is_train:
        filename = train_file_list[imgid]
        hand_label = train_hand_labels[imgid]
    else:
        filename = valid_file_list[imgid]
        hand_label = valid_hand_labels[imgid]

    if use_gray:
        image = cv2.imread(filename, 0)
    else:
        image = cv2.imread(filename, 1)
        if image.ndim == 2:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    
    #print(filename)
    image = cv2.resize(image, (im_width, im_height))
    
    if config.enable_color_jitter:
        randval = npr.randint(0,10)
        if randval <= 2:
            scale = npr.randint(50,125)/100.0
            image = image.astype(np.float32)
            image = image * scale
            image = np.clip(image, 0.0, 255.0)
                

    if config.enable_gaussian_noise:
        randval = npr.randint(0,10)
        if randval <= 2:
            mean = 0
            var = np.random.randint(0,10)+1.0
            image = image_processing.gaussian_noise(image,mean,var)

    if config.enable_blur:
        randval = npr.randint(0,10)
        if randval <= 2:
            ksize = npr.randint(1,10)
            for j in range(ksize):
                image = cv2.GaussianBlur(image,(3,3),0)

        
    if config.enable_black_border:
        size = im_width
        black_size = npr.randint(0,int(size*0.33))
        randval = npr.randint(0,20)
        if randval == 0:
            image[:,0:black_size] = 0
        elif randval == 1:
            image[:,(size-black_size):size] = 0
        elif randval == 2:
            image[0:black_size,:] = 0
        elif randval == 3:
            image[(size-black_size):size,:] = 0
    
    image = image.astype(np.float32)
    #image = (image - 127.5)*0.0078125
    if use_gray:
        image = image[:,:,np.newaxis]
    #print(image.shape)
    return (image, hand_label)

def _set_shapes(img, hand_label):
    img.set_shape([im_height,im_width,im_channel])
    hand_label.set_shape(class_num)
    return img,hand_label

def _get_dataset_pipeline(imgids, im_width, im_height, batch_size, epoch, buffer_size, thread_num, use_gray, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices(imgids)
    dataset.shuffle(buffer_size)
    dataset = dataset.map(
        lambda imgId: tf.py_func(
                func = _parse_data,
                inp = [imgId, is_train, im_width, im_height, use_gray],
                Tout = [tf.float32, tf.float32]),
        num_parallel_calls = thread_num
    )
    dataset = dataset.map(_set_shapes, num_parallel_calls = thread_num)
    dataset = dataset.batch(batch_size).repeat(epoch)
    dataset = dataset.prefetch(100)
    return dataset

# def get_train_dataset_pipeline(image_path, anno_file, im_w, im_h, batch_size, epoch, buffer_size, thread_num, use_gray):
def get_train_dataset_pipeline(anno_file, im_w, im_h, batch_size, epoch, buffer_size, thread_num, use_gray):
    global train_file_list, train_hand_labels, im_width, im_height, im_channel    
    # train_file_list, train_hand_labels, imgids = gen_data(image_path, anno_file)
    train_file_list, train_hand_labels, imgids = gen_data(anno_file)
    im_width, im_height = im_w, im_h

    if use_gray:
        im_channel = 1
    else:
        im_channel = 3
    return _get_dataset_pipeline(imgids, im_width, im_height, batch_size, epoch, buffer_size, thread_num, use_gray, True)

# def get_valid_dataset_pipeline(image_path, anno_file, im_w, im_h, batch_size, epoch, buffer_size, thread_num, use_gray):
def get_valid_dataset_pipeline(anno_file, im_w, im_h, batch_size, epoch, buffer_size, thread_num, use_gray):
    global valid_file_list, valid_hand_labels, im_width, im_height, im_channel    
    # valid_file_list, valid_hand_labels, imgids = gen_data(image_path, anno_file)
    valid_file_list, valid_hand_labels, imgids = gen_data(anno_file)
    im_width, im_height = im_w, im_h

    if use_gray:
        im_channel = 1
    else:
        im_channel = 3
    return _get_dataset_pipeline(imgids, im_width, im_height, batch_size, epoch, buffer_size, thread_num, use_gray, False)


