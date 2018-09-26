from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries,watershed
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np
import warnings
import cv2
import os
import sys
import argparse
from skimage import morphology
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import hough_circle, hough_circle_peaks
from sklearn.mixture import GMM
from imutils.object_detection import non_max_suppression
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
parser =  argparse.ArgumentParser()

parser.add_argument("-i","--img_dir" ,default ='img',help = "path to input image")
parser.add_argument("-m","--model_file" ,default ='RF_hu_model.sav',help = "path to the .sav color model")
parser.add_argument("-o","--out_file" ,default ='GCP_locations.csv',help = "path to the output .csv file")
parser.add_argument("-v","--vis" ,type=bool,default =False,help = "set it to true to visulaize results")
parser.add_argument("-cd","--c_data" ,default ='GCP_color_data.npy',help = "path to the .sav color data")
warnings.filterwarnings("ignore", category=DeprecationWarning)
args= parser.parse_args()
img_dir  =  args.img_dir
files = os.listdir(img_dir)
files.sort()
model_file = args.model_file
c_data_file = args.c_data
rgb_data = np.load(c_data_file)
out_filename = args.out_file
vis = args.vis

def check_similarity(e1,sc):
    fl = 'crop_pos'
    ff =  os.listdir(fl)
    for f in ff:
        ii= cv2.imread(os.path.join(fl,f),0)
        ii = cv2.copyMakeBorder(ii,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ii = cv2.resize(ii,(80,80),cv2.INTER_AREA)

        ii=  np.where(ii>128,255,0).astype('uint8')
        bin_img = np.where(ii >0,1,0).astype('uint8')
        skel1 = morphology.skeletonize(bin_img)
        skel1 = (skel1*255).astype('uint8')
        edges1 = canny(ii, sigma=3, low_threshold=10, high_threshold=100)
        edges1 = (edges1*255).astype('uint8')
        e2 =cv2.bitwise_or(skel1,edges1)
        _,cnts0, _ = cv2.findContours(e1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts0 = sorted(cnts0, key = cv2.contourArea, reverse = True)
        cnt0 = cnts0[0]
        _,cnts1, _ = cv2.findContours(e2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = sorted(cnts1, key = cv2.contourArea,reverse=True)
        cnt1 = cnts1[0]
        cnt0 = cnt0.astype(np.float64)
        cnt1 = cnt1.astype(np.float64)
        #print (cnt1)
        print (cnt0.shape)
        print (cnt1.shape)
        
        #sc.setBendingEnergyWeight(0.5)
        #sc.setShapeContextWeight(1.5)
        #print (sc.getImageAppearanceWeight())
        dist = 10000
        if abs(cnt1.shape[0]-cnt0.shape[0]) < 30:
        #sc.setImages(img1,img2)
            try:
                distance = sc.computeDistance(cnt0,cnt1)
                if distance<4 and distance>0:
            #        cn = cn+1
                    dist=distance
            except Exception as e: # and assert it really does
                if isinstance(e, AssertionError):
                    break
            if dist<10000:
                break
             #   print (distance)
             #   print ('match' + f + '_' +f2)
    return dist  
        

def create_model(features,ncomp=1):
    gmm = GMM(n_components=ncomp,covariance_type='diag')
    gmm.fit(features)
    return gmm


clr_model =create_model(rgb_data,ncomp=2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
shp_model = pickle.load(open(model_file,'rb'))
out_file =  open(out_filename,'w')
writer =  csv.writer(out_file)
sc = cv2.createShapeContextDistanceExtractor(nAngularBins = 12,nRadialBins = 5,innerRadius =0.125,outerRadius = 2)
sc.setRotationInvariant(True)
 
for f in files[2:]:
    ind=0
    if not (f.endswith('.JPG') or f.endswith('.jpg')):
        continue
    print (f)
    img  = cv2.imread(os.path.join(img_dir,f))
    h,w = img.shape[:2]
    img_vis = img.copy()
    #img_hsv =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    

    X = img.reshape(-1,3)
    prob  =  clr_model.score(X)
    prob_mat = prob.reshape(img.shape[:2])
    segm =  np.where(prob_mat>0.92*np.max(prob),255,0).astype('uint8')

    segm = cv2.morphologyEx(segm,cv2.MORPH_CLOSE,kernel,iterations=3)
    segm = np.where(segm>0,255,0).astype('uint8')
    
    _,cnts, _ = cv2.findContours(segm.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    cnts  =  filter(lambda x: (cv2.contourArea(x)<3000 and cv2.contourArea(x)>5),cnts)
    
    segm_temp = cv2.resize(segm,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
    rects =[] 
    for cnt in cnts:
     #   print ('r')
        rects.append(cv2.boundingRect(cnt))
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects if (w*h < 10000 and w*h>5)])
        
    #print (rects)
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    loc =[]
    segm_vis  = segm.copy()
    if len(pick)>0:
        marker = np.ones((segm.shape[:2]),dtype=np.int32)
        for x1,y1,x2,y2 in pick:
            
            cv2.rectangle(marker,(x1-20,y1-20),(x2+20,y2+20),0,-1)

        marker[segm==255]=2

        reg_grow = cv2.watershed(img,marker).astype('uint8')

        reg_grow =np.where(reg_grow==2,255,0).astype('uint8')
        reg_vis = cv2.resize(reg_grow,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
        #cv2.imwrite('reg_' + f,reg_grow) 
        reg_grow = cv2.morphologyEx(reg_grow,cv2.MORPH_OPEN,kernel2,iterations=1)
        if vis:
            cv2.imshow('tt',reg_vis)
            k= cv2.waitKey(0)

            if k == ord('e'):
                cv2.destroyAllWindows()
                sys.exit(-1)

        _,cnts, _ = cv2.findContours(reg_grow.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        cnts  =  filter(lambda x: (cv2.contourArea(x)<3000 and cv2.contourArea(x)>20),cnts)
        
        rects =[] 
        for cnt in cnts:
            rects.append(cv2.boundingRect(cnt))
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects if (w*h < 8000 and w*h>40)])
        
        reg_temp=reg_grow.copy()
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        h,w = img.shape[:2]
        ind =0
        for x1,y1,x2,y2 in pick:
                         
            cv2.rectangle(reg_temp,(x1,y1),(x2,y2),255,2)
       #     print (h,w)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            crop0 = reg_grow[max(0,y1-5):min(y2+5,h),max(0,x1-5):min(w,x2+5)]
            ch,cw = crop0.shape[:2]
            crop1 = img[max(0,y1-5):min(y2+5,h),max(0,x1-5):min(w,x2+5),:]
            s =max([ch,cw])
            crop = cv2.resize(crop0,(s,s),interpolation=cv2.INTER_AREA)
            
            if s<100:
            
                #moments = cv2.HuMoments(cv2.moments(crop)).flatten().reshape(1,-1)
                #pred = shp_model.predict(moments)
                                
                
                edges = canny(crop, sigma=3, low_threshold=10, high_threshold=100)
                edges = (edges*255).astype('uint8')
                if vis:
                    cv2.imshow('edged',edges)
                _,cnts0, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts0 = sorted(cnts0, key = cv2.contourArea, reverse = True)
                cnt0 = cnts0[0]
                area = cv2.contourArea(cnt0)   
                rect_area = (y2-y1)*(x2-x1)
                extent  = float(area)/rect_area
                peri = cv2.arcLength(cnt0, True)
                compactness = (4*3.14*area)/(peri*peri)
                #print ('compactness')
                #print (compactness)
                approx = cv2.approxPolyDP(cnt0, 0.04 * peri, True)
                vertices2 = (len(approx))
                #print (vertices2)
                #print (extent)
                bin_img = np.where(crop >128,1,0).astype('uint8')
                skel = morphology.skeletonize(bin_img)
                skel = (skel*255).astype('uint8')
                skel_or = skel.copy()
                e1 = cv2.bitwise_or(skel,edges)
                _,cnts0, _ = cv2.findContours(skel.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts0 = sorted(cnts0, key = cv2.contourArea, reverse = True)
                cnt0 = cnts0[0]
                     
                peri = cv2.arcLength(cnt0, True)
                approx = cv2.approxPolyDP(cnt0, 0.04 * peri, True)
                vertices = (len(approx))
                hs,theta,d = hough_line(skel)
                angles= []
                cn =0
                votes =[]
                rho =[]
                temp  =  skel.copy()
                for out, angle, dist in zip(*hough_line_peaks(hs, theta, d,min_angle=80,num_peaks=2)):
                    rho.append(dist)
                    votes.append(out)
                    angles.append(angle)
                    if out>10:
                        cn =  cn+1
                    
                    a = np.cos(angle)
                    b = np.sin(angle)
                    x0 = a*dist
                    y0 = b*dist
                    x01 = int(x0 + 1000*(-b))
                    y01 = int(y0 + 1000*(a))
                    x02 = int(x0 - 1000*(-b))
                    y02 = int(y0 - 1000*(a))
                    #x1,y1,x2,y2 = line[0]
                    cv2.line(temp,(x01,y01),(x02,y02),255,2)
                    if vis:
                        cv2.imshow('ss',temp)
                        
                        k=cv2.waitKey(0)
                        if k==ord('e'):
                            cv2.destroyAllWindows()
                            sys.exit(-1)
                #if len(angles)>1:
                    #print (abs(angles[0] - angles[1]))
                    
                if len(votes)>=2:
                    #print (vertices)
                    ratio =  votes[1]/(votes[0]*1.0)
                #    n_ratio = arm_lengths[1]/(arm_lengths[0]*1.0)
                    #print ('ratio')
                 #   print (n_ratio)
                if cn==2:
                    dist = check_similarity(e1,sc)
                    print (dist)
                    
                    if (vertices>=4 and vertices<=7) and (abs(angles[0] - angles[1])>1.2 and abs(angles[0] - angles[1])<1.7) and (ratio>0.7 and ratio<1.15) and (vertices2>4 and vertices2<=6) and dist<4 :
                        #print ('present')
                        #cv2.imwrite('skel_r_' +f,skel)
                        if vis:
                            cv2.imshow('out',crop)
                        #cv2.imwrite('skel_' +f,skel_or)
                        A = np.array([[np.cos(angles[0]), np.sin(angles[0])],[np.cos(angles[1]), np.sin(angles[1])]])
                        b = np.array([[rho[0]], [rho[1]]])
                        x0, y0 = np.linalg.solve(A, b)
                        #x0, y0 = int(np.round(x0)), int(np.round(y0))
                        print (x1+x0[0],y1+y0[0])
                        loc.append([x1+x0[0],y1+y0[0]])
                        cv2.rectangle(img_vis,(x1,y1),(x2,y2),(0,0,0),3)
                        k = cv2.waitKey(0)
                        if k == ord('e'):
                            cv2.destroyAllWindows()
                            sys.exit(-1)
        if vis:
            img_vis = cv2.resize(img_vis,None,fx=0.3,fy=0.3,interpolation=cv2.INTER_AREA)
            cv2.imshow('re',img_vis)
        #if len(loc) >0:
    content = [f,loc]        
    writer.writerow(content)

