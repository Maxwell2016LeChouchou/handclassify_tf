import numpy as np
import cv2
import threading
import argparse
import math
import os,sys
import numpy.random as npr
sys.path.append(os.getcwd())
from config import config
import tools.image_processing as image_processing

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.hand_names, self.hand_nums = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.hand_names, self.hand_nums
        except Exception:
            return None

def gen_hand_minibatch_thread(size, start_idx, annotation_lines, imdir, hand_save_dir, base_nums, class_num):
    num_images = len(annotation_lines)
    hand_names = list()
    hand_nums = list()
    for i in range(class_num):
        hand_nums.append(0)
    for i in range(num_images):
        cur_annotation_line = annotation_lines[i].strip().split()
        im_path = cur_annotation_line[0]
        cur_annotation = cur_annotation_line[1:]
        img = cv2.imread(os.path.join(imdir, im_path),0)
        cur_hand_names, cur_hand_nums = gen_hand_for_one_image(size, start_idx+i, img, \
            hand_save_dir, cur_annotation, base_nums, class_num)
        hand_names = hand_names + cur_hand_names
        for j in range(class_num):
            hand_nums[j] += cur_hand_nums[j]

    return hand_names, hand_nums


def gen_hand_minibatch(size, start_idx, annotation_lines, imdir, hand_save_dir, base_nums, thread_num, class_num):
    num_images = len(annotation_lines)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    threads = []
    for t in range(thread_num):
        cur_start_idx = int(num_per_thread*t)
        cur_end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_annotation_lines = annotation_lines[cur_start_idx:cur_end_idx]
        cur_thread = MyThread(gen_hand_minibatch_thread,(size, start_idx+cur_start_idx, cur_annotation_lines,
                                                        imdir, hand_save_dir, base_nums, class_num))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    hand_names = list()
    hand_nums = list()
    for i in range(class_num):
        hand_nums.append(0)
    for t in range(thread_num):
        cur_hand_names, cur_hand_nums = threads[t].get_result()
        hand_names = hand_names + cur_hand_names
        for j in range(class_num):
            hand_nums[j] += cur_hand_nums[j]

    return hand_names, hand_nums
	

def gen_hand_for_one_image(size, idx, ori_img, save_dir, annotation, base_nums, class_num):
    hand_names = list()
    hand_nums = list()
    for i in range(class_num):
        hand_nums.append(0)

    #print(annotation)
    width = ori_img.shape[1]
    height = ori_img.shape[0]
    x1 = float(annotation[0])
    y1 = float(annotation[1])
    # w = float(annotation[2])
    # h = float(annotation[3])
    x2 = float(annotation[2])
    y2 = float(annotation[3])
    w = x2 - x1
    h = y2 - y1

    label = int(annotation[4])
    cx = x1+0.5*w
    cy = y1+0.5*h

    if label < 0 or label >= class_num:
        return hand_names, hand_nums

    
    bbox_size = max(w,h)
    w = bbox_size
    h = bbox_size
 
    border_size = int(0.5*bbox_size)
    img = cv2.copyMakeBorder(ori_img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, (0,0,0))
    cx += border_size
    cy += border_size
    
    init_rot = 0
    
    base_num = base_nums[label]

    try_num = 0
    hand_num = 0
    while hand_num < base_num:
        try_num += 1
        if try_num > base_num*100:
            break
        cur_angle = npr.randint(int(config.min_rot_angle - init_rot),int(config.max_rot_angle - init_rot)+1)
        
        rot_img = image_processing.rotateOnlyImage(img, cx,cy,cur_angle,1)

        cur_size = int(npr.randint(110, 136)*0.01*bbox_size)
        
        # delta here is the offset of box center
        delta_x = npr.randint(-int(w * 0.15), int(w * 0.15)+1)
        delta_y = npr.randint(-int(h * 0.15), int(h * 0.15)+1)
        
        nx1 = max(int(cx + delta_x - cur_size/2), 0)
        ny1 = max(int(cy + delta_y - cur_size/2), 0)
        nx2 = nx1 + cur_size
        ny2 = ny1 + cur_size

        if nx2 > rot_img.shape[1] or ny2 > rot_img.shape[0]:
            continue
        
        cropped_im = rot_img[ny1 : ny2, nx1 : nx2]
        resized_im = cv2.resize(cropped_im, (size, size), interpolation=cv2.INTER_LINEAR)
        
        out_label = label
        if config.hand_random_flip:
            flip_val = npr.randint(0,2)
            if flip_val == 1:
                resized_im = resized_im[:,::-1]
                # if label == 5:
                #     out_label = 7
                # elif label == 7:
                #     out_label = 5
        
        save_file = '%s/%d_%d.jpg'%(save_dir,idx,hand_num)
        if cv2.imwrite(save_file, resized_im):
            line = '%s/%d_%d.jpg %d'%(save_dir,idx,hand_num,out_label)
            hand_names.append(line)
            hand_nums[out_label] += 1
            hand_num += 1

    return hand_names, hand_nums

def gen_hand(image_path, anno_file, size=20, base_num = 1, thread_num = 4, class_num = 8):
    save_dir = './prepare_data/%d'%(size)
    hand_save_dir = save_dir + '/hand'
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(hand_save_dir):
        os.mkdir(hand_save_dir)



    f = open(anno_file, 'r')
    annotation_lines = f.readlines()
    f.close()

    f = open(os.path.join(save_dir, 'hand.txt'), 'w')    
    num = len(annotation_lines)
    print("%d pics in total" % num)

    total_label_num = 0
    label_nums = list()
    for i in range(class_num):
        label_nums.append(0)

    for i in range(num):
        cur_line = annotation_lines[i].strip()
        cur_splits = cur_line.split()
        label = int(cur_splits[5])
        if label >= 0 and label < class_num:
            label_nums[label] = label_nums[label] + 1
            total_label_num += 1

    label_ratios = list()
    for i in range(class_num):
        label_ratios.append(max(1e-6,float(label_nums[i]) / total_label_num))
        print('ratio %d : %f'%(i, label_ratios[i]))
   
    base_nums = list()
    for i in range(class_num):
        cur_base_num = int(max(1.0, base_num * 1.0/class_num / label_ratios[i]))
        base_nums.append(cur_base_num)
        print('base_num %d : %d'%(i, cur_base_num))

    batch_size = thread_num*10
    total_hand_num = 0
    hand_nums = list()
    for i in range(class_num):
        hand_nums.append(0)
    start_idx = 0
    while start_idx < num:
        end_idx = min(start_idx+batch_size,num)
        cur_annotation_lines = annotation_lines[start_idx:end_idx]
        hand_names, cur_hand_nums = gen_hand_minibatch(size, start_idx, cur_annotation_lines, \
                                            image_path, hand_save_dir, base_nums, thread_num, class_num)
        cur_num = len(hand_names)
        for i in range(cur_num):
            f.write(hand_names[i]+'\n')
        total_hand_num += cur_num
        for i in range(class_num):
            hand_nums[i] += cur_hand_nums[i]
        start_idx = end_idx
        show_line = '%s images done, total: %d'%(start_idx, total_hand_num)
        for i in range(class_num):
            show_line = show_line + ', [%d]:%d'%(i,hand_nums[i])
        print(show_line)

    f.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', dest='image_path', help='image path',
                        default='data/untouch', type=str)
    parser.add_argument('--anno_file', dest='anno_file', help='anno file',
                        default='data/untouch/anno_hand_9class.txt', type=str)
    parser.add_argument('--size', dest='size', help='112, 96, 80, 64', default='96', type=str)
    parser.add_argument('--base_num', dest='base_num', help='base num', default='10', type=str)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num', default='4', type=str)
    parser.add_argument('--class_num', dest='class_num', help='class num', default='9', type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with argument:')
    print(args)
    gen_hand(args.image_path, args.anno_file, int(args.size), int(args.base_num), int(args.thread_num), int(args.class_num))

