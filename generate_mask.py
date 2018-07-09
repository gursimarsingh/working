from PIL import Image, ImageDraw
import numpy as np
import scipy.misc
import cv2

BODY_PARTS = {"Nose":0, "Neck":1, "RShoulder":2,  "RElbow":3, "RWrist":4,
                        "LShoulder":5,  "LElbow":6, "LWrist":7, "RHip":8, "RKnee":9,
                         "RAnkle":10, "LHip":11, "LKnee"    :12, "LAnkle":13, "REye":14,
                          "LEye":15, "REar":16, "LEar":17, "Bkg":18} ### 1,15,16,17,18 ####
import os
import json
from glob import glob
import argparse
from tqdm import tqdm, trange
from time import sleep
import csv
import pdb
from skimage.draw import circle, line_aa, polygon
from skimage.morphology import dilation, erosion, square
import skimage.io


class GenerateMask:

     def __init__(self):
        print "nothing as per now"

     def produce_ma_mask(self, kp_array, img_size, point_radius=15):
       	
        MISSING_VALUE = 0
        mask = np.zeros(shape=img_size, dtype=np.uint8)
        limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8],  [9,10],
                  [10,11],  [12,13], [13,14], [2,1], [1,15], [15,17],
                   [1,16], [16,18], [2,17], [2,18], [9,12], [17,18]] #[12,6], [9,3],[6,9],[3,12]
        limbs = np.array(limbs) - 1
        #pdb.set_trace()
        for f, t in limbs:
            from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
            to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
    	#from_missing = kp_array[f][2] == MISSING_VALUE
    	#to_missing = kp_array[t][2] == MISSING_VALUE
            if from_missing or to_missing:
                continue

            norm_vec = kp_array[f] - kp_array[t]
            norm_vec = np.array([-norm_vec[1], norm_vec[0]])
            norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


            vetexes = np.array([
                kp_array[f] + norm_vec,
                kp_array[f] - norm_vec,
                kp_array[t] - norm_vec,
                kp_array[t] + norm_vec
            ])
    	#pdb.set_trace()
            yy, xx = polygon(vetexes[:, 1], vetexes[:, 0], shape=img_size)
            mask[yy, xx] = 255 

        for i, joint in enumerate(kp_array):
            if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            #if kp_array[i][2] == MISSING_VALUE:
                continue
            yy, xx = circle(joint[1], joint[0], radius=point_radius, shape=img_size)
            mask[yy, xx] = 255
        
        #mask[:,:] = np.where(mask[:,:] == True, 255, 0)
        #cv2.imshow('s',mask.astype(np.uint8))
        return np.uint8(mask)

     def segmentation(self,pose_coord,(h,w)):
        ### sure foreground
        mask1=self.produce_ma_mask(pose_coord, (h, w),5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        sure_fg=cv2.morphologyEx(np.uint8(mask1),cv2.MORPH_CLOSE,kernel, iterations = 1)	
            #mask = dilation(mask, square(5))
            #mask = erosion(mask, square(3))
            #print(mask)
        ### sure background
        kernel2=  np.ones((3,3),np.uint8)
        mask2=self.produce_ma_mask(pose_coord, (h, w),16)
        
        pos_bg = cv2.morphologyEx(np.uint8(mask2),cv2.MORPH_CLOSE,kernel, iterations = 3)
        pos_bg = cv2.dilate(pos_bg,kernel2,iterations = 3)
            
        return pos_bg,sure_fg

     def mask_generate(self, keypt_list, height, width,input_image = None):

        # need to do some preprocessing on keypt_list
        new_pose = np.array(keypt_list).reshape(18,3)
        pose_coord = new_pose[:, 0:2]
        #print pose_coord
        #print type(pose_coord)
        #print 'height, width, img_shape' , height, width, img_shape
        pos_bg,sure_fg= self.segmentation(pose_coord,(height,width))
        #mmm = self.produce_ma_mask(pose_coord, (height, width),flag)
#.astype(float)[..., np.newaxis].repeat(3, axis=-1)
        #self.img = np.zeros(shape= (height, width), dtype=np.uint8)
        '''for c in range(3):'''
        #self.img[:,:] = np.where(mmm[:,:] == True, 255, 0)
        self.bg=pos_bg
        self.fg=sure_fg
        return self.bg,self.fg
