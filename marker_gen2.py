import cv2 
import numpy as np
import os
import generate_mask as gm

folder  = ''
csv = ''
src =''


files= os.listdir(folder)
files.sort()

def decode_seed(seed):

	mask_init = np.zeros(seed.shape,dtype='uint8')
	mask_init  = np.where(seed==0,0,255).astype('uint8')

	return mask_init[:,:,0]

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

def add_limb(kp1,kp2,mask,point_radius=10):
	fill =255
	flag =1
	MISSING_VALUE =0
	from_missing = kp1[0] == MISSING_VALUE or kp1[1] == MISSING_VALUE
	to_missing = kp2[0] == MISSING_VALUE or kp2[1] == MISSING_VALUE
		#from_missing = kp1[2] == MISSING_VALUE
		#to_missing = kp2[2] == MISSING_VALUE
	if from_missing or to_missing:
		flag =0
	    return mask,flag
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

	return mask,flag

def ret_mid_pt(kp):
	bck = np.asarray([0,0])
	i=0
	
	for p in kp:
		if p[0]!=0 and p[1]!=0:
			bck = bck+np.asarray(p[0:2])
			i=i+1
	bck = bck/i
	#kp.append(list(bck.astype(np.uint8)))
	return list(bck.astype(np.uint8))

def getSkletetonMask(g,kp,h,w):
	bg,sure_fg = g.mask_generate(kp,h,w)
	
	
	#mid_hip = ret_mid_pt([kp[2],kp[3]])
	mid_hip = ret_mid_pt([kp[8],kp[11]])

	sure_fg,flag = add_limb(kp[1],mid_hip,sure_fg)
	if flag ==0:
		sure_fg,_ = add_limb(kp[8],kp[2],sure_fg,point_radius=5)
		sure_fg,_ = add_limb(kp[5],kp[11],sure_fg,point_radius=5)


	bg = add_limb(kp[1],mid_hip,bg,point_radius=30)
	#bg = add_limb(kp[1],kp[11],bg)
	bg = add_limb(kp[8],kp[2],bg,point_radius=14)
	bg = add_limb(kp[5],kp[11],bg,point_radius=14)
	bg = cv2.morphologyEx(bg,cv2.MORPH_DILATE,kernel,iterations=5)

	return bg,sure_fg

def create_marker(bg,fg):

	mark_im = np.zeros(im.shape,dtype='uint8')
	mark_im[:,:,0] = np.where(fg[:,:]==255,255,0)
	mark_im[:,:,2] = np.where(bg[:,:]==0,255,0)

	return mark_im

g=gm.GenerateMask()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

for filename in files:
	filename = '001_3528-8_70514.jpg'

	seed  = cv2.imread(os.path.join(folder,filename))
	im = cv2.imread(os.path.join(src,filename))

	mask_init =  decode_seed(seed)

	h,w,d=im.shape
	kp=read_csv(filename) 
	sure_bg,sure_fg = getSkletetonMask(g,kp,h,w)

	mask_init[sure_fg==255] = 255
	mask_init[sure_bg== 0 ] =  0

	mark_im = create_marker(sure_bg,mask_init)
	cv2.imshow('mm',mark_im)
	cv.imshow('sk',sure_fg)



	