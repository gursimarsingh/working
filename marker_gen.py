from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.draw import circle, line_aa, polygon
from skimage import filters
from sklearn import metrics, preprocessing
from sklearn import pipeline, cluster, mixture, decomposition
import numpy as np
#import pycoop, pycoop.potentials as potentials
#from  raw_read import get_mask
import cv2 
import numpy as np
import generate_mask as gm
import os 
import argparse

img_src = '../images'
csv_src  = '../csv'
dst = '../export_mask'
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True, help = "Path to the image")
# args = vars(ap.parse_args())	
file_list = os.listdir(img_src)
file_list.sort()

def read_csv(filename):
	
	f=filename.split('.')
	with open(csv_src  + f[0]+'.csv', 'rb') as csvfile:
		pt=[]
		for line in csvfile.readlines():	
			array = line.split(',')
		for k in range(len(array)):
			if k%3==0:
				x=int(round(float(array[k])*w))
				y=int(round(float(array[k+1])*h))
				
				pt.append([x,y,1])

	return pt
# def read_csv(filename):
	
# 	f=filename.split('.')
# 	with open(csv_src  + f[0]+'.csv', 'rb') as csvfile:
# 		pt=[]
# 		lines = csvfile.readlines()
# 		x_co =  lines[0].split(',')
# 		y_co =  lines[1].split(',')
# 		for x,y in zip(x_co,y_co):
# 			print x,y
# 			pt.append([int(round(float(x))),int(round(float(y))),1])


# 	return pt

def create_rect(pt,h,w):

	rect=[]
	sort_x= sorted(pt,key= lambda x:x[0])
	sort_y= sorted(pt,key= lambda y:y[1])
	for item in sort_x:
		if item[0]!=0 and item[1]!=0:
			rect.append(item[0]-int(0.1*w))
			break

	for item in sort_y:
		if item[0]!=0 and item[1]!=0:
			rect.append(item[1]-int(0.1*h))
			break
	#print rect
	rect.append(sort_x[-1][0] +int(0.1*w))
	rect.append(sort_y[-1][1] +int(0.1*h))
	
	if rect[0]<0:
		rect[0]=0
	if rect[1]<0:
		rect[1]=0

	if rect[2]>w:
		rect[2]=w
	if rect[3]>h:
		rect[3]=h

	return rect
def add_limb(kp1,kp2,mask,point_radius=10,flag=1):
	if flag==1:
		fill =255
	else:
		fill =0
	MISSING_VALUE =0
	from_missing = kp1[0] == MISSING_VALUE or kp1[1] == MISSING_VALUE
	to_missing = kp2[0] == MISSING_VALUE or kp2[1] == MISSING_VALUE
		#from_missing = kp1[2] == MISSING_VALUE
		#to_missing = kp2[2] == MISSING_VALUE
	if from_missing or to_missing:
	    return mask
	img_size = (h,w)
	kp1 = np.asarray(kp1[0:2])
	kp2 = np.asarray(kp2[0:2])
	norm_vec = kp1 - kp2
	norm_vec = np.array([-norm_vec[1],norm_vec[0]])
	norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


	vetexes = np.array([
	    kp1 + norm_vec,
	    kp1 - norm_vec,
	    kp2 - norm_vec,
	    kp2 + norm_vec
	])
#pdb.set_trace()
	yy, xx = polygon(vetexes[:, 1], vetexes[:, 0], shape=img_size)
	mask[yy, xx] = fill

	yy, xx = circle(kp1[1], kp1[0], radius=point_radius, shape=img_size)
	mask[yy, xx] = fill
	yy, xx = circle(kp2[1], kp2[0], radius=point_radius, shape=img_size)
	mask[yy, xx] = fill

	return mask
def ret_mid_pt(kp):
	# rsh = np.asarray(kp[2]) 
	# lsh = np.asarray(kp[5])
	# lhip = np.asarray(kp[11])
	# rhip= np.asarray(kp[8])
	# pt = [rsh[0:2],lsh[0:2],lhip[0:2],rhip[0:2]]
	bck = np.asarray([0,0])
	i=0
	for p in kp:
		if p[0]!=0 and p[1]!=0:
			bck = bck+np.asarray(p[0:2])
			i=i+1
	bck = bck/i
	#kp.append(list(bck.astype(np.uint8)))
	return list(bck.astype(np.uint8))


g=gm.GenerateMask()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
for filename in file_list:
	img_src = ''
	filename = '001_3528-8_70514.jpg'
	dst =''
	csv_src = ''
# load the image and convert it to a floating point data type
	image = img_as_float(io.imread(os.path.join(img_src,filename)))

	img = cv2.imread(os.path.join(img_src,filename))
	#image =  cv2.resize(image,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
	#image =  cv2.resize(image,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
	#img =  cv2.resize(img,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC).astype(np.uint8)
	temp = image.copy().astype('uint8')
	h,w,d=image.shape

	

	kp=read_csv(filename) 
	bg,sure_fg = g.mask_generate(kp,h,w)
	
	
	#mid_hip = ret_mid_pt([kp[2],kp[3]])
	mid_hip = ret_mid_pt([kp[8],kp[11]])
	sure_fg = add_limb(kp[1],mid_hip,sure_fg)
	bg = add_limb(kp[1],mid_hip,bg,point_radius=30)
	#bg = add_limb(kp[1],kp[11],bg)
	bg = add_limb(kp[8],kp[2],bg,point_radius=14)
	bg = add_limb(kp[5],kp[11],bg,point_radius=14)
	bg = cv2.morphologyEx(bg,cv2.MORPH_DILATE,kernel,iterations=5)
	# bg = add_limb(kp[1],kp[11],bg)
	# bg = add_limb(kp[1],kp[11],bg)

	mask = np.zeros(img.shape,dtype='uint8')
	
	mask[:,:,0] = 255
	mask[:,:,1] = np.where(sure_fg[:,:]==255,0,255)
	mask[:,:,2] = np.where(sure_fg[:,:]==255,0,255)
	mask[:,:,0] = np.where(bg[:,:]==0,0,mask[:,:,0])
	mask[:,:,1] = np.where(bg[:,:]==0,0,mask[:,:,1])
	mask[:,:,2] = np.where(bg[:,:]==0,255,mask[:,:,2])

	#sure_fg2 = add_limb(kp[1],kp[11],sure_fg)
	# # sure_fg2 = add_limb(kp[1],kp[8],sure_fg2)
	# sure_fg = add_limb(kp[5],kp[8],sure_fg)
	# sure_fg = add_limb(kp[2],kp[11],sure_fg)
	# # sure_fg2 = add_limb(kp[2],kp[8],sure_fg2)
	# # sure_fg2 = add_limb(kp[5],kp[11],sure_fg2)

	
	
	cv2.imshow('re',mask)
	cv2.imwrite(os.path.join(dst,filename.split('.')[0]+'.png'),mask)
	#cv2.imshow('mask222',bg)
	#plt.axis("off")
	cv2.waitKey(0)

	# show the plots
#plt.show()