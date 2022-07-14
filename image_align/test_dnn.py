# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:33:55 2022

@author: Yushan
"""

import os
import math
import numpy as np
import cv2
import dlib

from imutils import face_utils

directory = '../images/White/Naomi Scott/'
out_dir = '../images/aligned_dcnn/'
img_name = 'aligned_img'
predictor = dlib.shape_predictor("../landmark_file/shape_predictor_68_face_landmarks.dat")


def dnn_detector(modelFile = "../model/dnn_model.caffemodel", configFile = "../model/deploy.prototxt"):
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def find_faces(net, image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (240, 240), (104.0, 117.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
     
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.8:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append([startX, startY, endX, endY])
            
    return faces
    
def detect_landmark(startX, startY, endX, endY, image):
    shape = predictor(image,dlib.rectangle(startX, startY, endX, endY))
    shape = face_utils.shape_to_np(shape)
    
    left_eye_center = shape[36:42].mean(axis=0).astype('int')
    right_eye_center = shape[43:48].mean(axis=0).astype('int')
    
    return left_eye_center, right_eye_center

def align_img(img, l_eye_center, r_eye_center):
    x1,y1 = r_eye_center
    x2,y2 = l_eye_center
    a=abs(y1-y2)
    b=abs(x2-x1)
    c=math.sqrt(a**2 +b**2)
    cos_alpha = np.arccos((b**2+c**2-a**2)/(2*b*c)) #radian
    alpha = (cos_alpha*180)/math.pi
    
    if(y1<y2):
        alpha = -alpha
    
    center = (img.shape[1]/2, img.shape[0]/2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle=alpha, scale=1.25)
    rotated_image = cv2.warpAffine(src=img, M=rot_matrix, dsize=(img.shape[1], img.shape[0]))
    
    return rotated_image

    
def preprocess(faces,image):
    for b,face in enumerate(faces):
        # name_img = img_name +str(a)
        startX, startY, endX, endY = [face[i] for i in range(len(face))]
        
        left_eye_center, right_eye_center = detect_landmark(startX, startY, endX, endY, image)
        
        # if(len(face)>1):
        #     name_img = name_img+'_'+str(b)
        
        startY = startY-30
        if(startY<0):
            startY=0
            
        endY=endY+30
        if(endY>image.shape[0]):
            endY=image.shape[0]
        
        startX=startX-30
        if(startX<0):
            startX=0
        
        endX=endX+30
        if(endX>image.shape[1]):
            endX=image.shape[1]
        
        crop_img = image[startY:endY, startX:endX]
        img_align = align_img(crop_img, left_eye_center, right_eye_center)
        # cv2.imwrite(os.path.join(out_dir,f'{name_img}.jpg'),img_align)
        return img_align
    
# def normalization(data):
#     return (data - np.min(data)) / ((np.max(data) - np.min(data)))

def normalization(data):
    normalized_data = data/np.linalg.norm(data)
    return normalized_data
        
        
# if __name__ =='__main__':
#     for a,img in enumerate(os.listdir(directory)):
#         img_dir = os.path.join(directory, img)
#         image = cv2.imread(img_dir)
        
#         net = dnn_detector()
#         faces = find_faces(net, image)
#         preprocess(faces, a)