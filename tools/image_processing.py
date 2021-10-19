import numpy as np
from numpy import *
import cv2
import math
import sys,os
sys.path.append(os.getcwd())
from config import config

def transform(im, train = False):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :return: [batch, channel, height, width]
    """
    scale = np.random.randint(3,7) * 0.2
    im = im * scale
    im = np.clip(im,0.0,255.0)

    if train:
        if config.enable_gaussian_noise:
            mean = 0
            var = np.random.randint(0,10)+1.0
            im = gaussian_noise(im,mean,var)
            

    im_tensor = im[np.newaxis, np.newaxis, :]
    im_tensor = (im_tensor - 127.5)*0.0078125
    return im_tensor

def gaussian_noise(image, mean=0, var=0.001):
    tmp_image = np.array(image, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = tmp_image + noise
    out = np.clip(out, 0.0, 255.0)
    return out

def rotateOnlyImage(image, cx, cy, angle, scale):
    if angle == 0:
        rot_image = image.copy()
        return rot_image
    else:
        w = image.shape[1]
        h = image.shape[0]
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
   
        rot_image = cv2.warpAffine(image,M,(w,h))
        return rot_image

def rotateWithLandmark(image, landmark, angle, scale):
    if angle == 0:
        rot_image = image.copy()
        landmark1 = landmark.copy()
        return rot_image,landmark1
    else:
        w = image.shape[1]
        h = image.shape[0]
        #cx = landmark[4]
        #cy = landmark[5]
        cx = 0.25*(landmark[0]+landmark[2]+landmark[6]+landmark[8])
        cy = 0.25*(landmark[1]+landmark[3]+landmark[7]+landmark[9])
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
    
        in_coords = np.array([[landmark[0], landmark[2], landmark[4], landmark[6], landmark[8]], 
                              [landmark[1], landmark[3], landmark[5], landmark[7], landmark[9]], 
                              [1,1,1,1,1]], dtype=np.float32)
    
        #rotate
  
        rot_image = cv2.warpAffine(image,M,(w,h))
        out_coords = np.dot(M,in_coords)
        landmark1 = np.array(landmark,dtype=np.float32).copy()
        for i in range(5):
            landmark1[i*2] = out_coords[0][i]
            landmark1[i*2+1] = out_coords[1][i]
        return rot_image, landmark1
		
def rotateWithLandmark106(image, cx,cy,landmark_x,landmark_y, angle, scale):
    if angle == 0:
        rot_image = image.copy()
        landmark_x1 = landmark_x.copy()
        landmark_y1 = landmark_y.copy()
        return rot_image,landmark_x1,landmark_y1
    else:
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
        #rotate
  
        rot_image = cv2.warpAffine(image,M,(w,h))
        landmark_x1 = np.array(landmark_x,dtype=np.float32).copy()
        landmark_y1 = np.array(landmark_y,dtype=np.float32).copy()
        for i in range(106):
            landmark_x1[i] = M[0][0]*landmark_x[i]+M[0][1]*landmark_y1[i]+M[0][2]
            landmark_y1[i] = M[1][0]*landmark_x[i]+M[1][1]*landmark_y1[i]+M[1][2]
        return rot_image, landmark_x1,landmark_y1
		
def rotateWithLandmark17(image, cx,cy,landmark_x,landmark_y, angle, scale):
    if angle == 0:
        rot_image = image.copy()
        landmark_x1 = landmark_x.copy()
        landmark_y1 = landmark_y.copy()
        return rot_image,landmark_x1,landmark_y1
    else:
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
        #rotate
  
        rot_image = cv2.warpAffine(image,M,(w,h))
        landmark_x1 = np.array(landmark_x,dtype=np.float32).copy()
        landmark_y1 = np.array(landmark_y,dtype=np.float32).copy()
        for i in range(17):
            landmark_x1[i] = M[0][0]*landmark_x[i]+M[0][1]*landmark_y1[i]+M[0][2]
            landmark_y1[i] = M[1][0]*landmark_x[i]+M[1][1]*landmark_y1[i]+M[1][2]
        return rot_image, landmark_x1,landmark_y1

def rotateLandmark(landmark, angle, scale):
    if angle == 0:
        landmark1 = landmark.copy()
        return landmark1
    else:
        #cx = landmark[4]
        #cy = landmark[5]
        cx = 0.25*(landmark[0]+landmark[2]+landmark[6]+landmark[8])
        cy = 0.25*(landmark[1]+landmark[3]+landmark[7]+landmark[9])
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
    
        in_coords = np.array([[landmark[0], landmark[2], landmark[4], landmark[6], landmark[8]], 
                              [landmark[1], landmark[3], landmark[5], landmark[7], landmark[9]], 
                              [1,1,1,1,1]], dtype=np.float32)
    
        #rotate
  
        out_coords = np.dot(M,in_coords)
        landmark1 = np.array(landmark,dtype=np.float32).copy()
        for i in range(5):
            landmark1[i*2] = out_coords[0][i]
            landmark1[i*2+1] = out_coords[1][i]
        return landmark1
		
def rotateLandmark106(cx,cy,landmark_x,landmark_y, angle, scale):
    if angle == 0:
        landmark_x1 = landmark_x.copy()
        landmark_y1 = landmark_y.copy()
        return landmark_x1,landmark_y1
    else:
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
        #rotate
        landmark_x1 = np.array(landmark_x,dtype=np.float32).copy()
        landmark_y1 = np.array(landmark_y,dtype=np.float32).copy()
        for i in range(106):
            landmark_x1[i] = M[0][0]*landmark_x[i]+M[0][1]*landmark_y1[i]+M[0][2]
            landmark_y1[i] = M[1][0]*landmark_x[i]+M[1][1]*landmark_y1[i]+M[1][2]
        return landmark_x1,landmark_y1
		
def rotateLandmark17(cx,cy,landmark_x,landmark_y, angle, scale):
    if angle == 0:
        landmark_x1 = landmark_x.copy()
        landmark_y1 = landmark_y.copy()
        return landmark_x1,landmark_y1
    else:
        #rotate matrix
        M = cv2.getRotationMatrix2D((cx,cy), angle, scale)
    
        #rotate
        landmark_x1 = np.array(landmark_x,dtype=np.float32).copy()
        landmark_y1 = np.array(landmark_y,dtype=np.float32).copy()
        for i in range(17):
            landmark_x1[i] = M[0][0]*landmark_x[i]+M[0][1]*landmark_y1[i]+M[0][2]
            landmark_y1[i] = M[1][0]*landmark_x[i]+M[1][1]*landmark_y1[i]+M[1][2]
        return landmark_x1,landmark_y1
