#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 15:40:19 2022

@author: usn
"""
import numpy as np
import cv2
import os

from xlwt import Workbook
from backbone.model_irse import IR_50
from util.extract_feature_v2 import extract_feature

from faiss_indexer.faiss_indexer_intern import FaceIndexer
from image_align.test_dnn import dnn_detector, find_faces, preprocess, normalization

data_root = '../images/White/'
test_img_path = '../images/sample_imgs/aligned_img6.jpg'
model_root = '../model/FaceEvolve/backbone_ir50_ms1m_epoch63.pth'

backbone = IR_50([112,112])
train_img_count = 4

store_list = []

class face_service:
    def __init__(self):
        self.indexer = FaceIndexer('indexer/')
        
    def search(self, emb_array, k=1):
        dists, indices, uids = self.indexer.search(emb_array, k)
        profile_dicts = []
        thresh = 0.4
        for i in range(len(dists)):
            profile_dict = {}
            for j in range(len(dists[i])):
                if dists[i][j] >= thresh:
                    uid = uids[i][j]
                    score = dists[i][j]
                    if score > profile_dict.get(uid, 0):
                        profile_dict[uid] = score
            profile_dicts.append(profile_dict)
        return profile_dicts
    
    def train(self):
        net = dnn_detector()
        
        for name in os.listdir(data_root):
            img_lists = []
            name_dir = os.path.join(data_root , name)
            for img_file in os.listdir(name_dir):
                img_file_dir = os.path.join(name_dir, img_file)
                image = cv2.imread(img_file_dir)
                faces = find_faces(net, image)
                img_align = preprocess(faces,image)
                img_lists.append(img_align)
        
            for i in range(0,train_img_count):
                test_dict = {}
                img_file_dir = os.path.join(name_dir, img_file)
                
                emb = extract_feature(img_lists[0],
                                            backbone,
                                            model_root)
                emb = emb[0].numpy()
                emb = normalization(emb)
                
                img_lists.pop(0)
                
                test_dict[name]=[emb]
                self.indexer.enroll(test_dict)
                
            store_list.append({name:img_lists})
            print("Generating image features ..................")
            
                
if __name__ =='__main__':
    face_obj = face_service()
    face_obj.train()
    
    wb = Workbook()
    sheet = wb.add_sheet('sheet 1', cell_overwrite_ok=True)
    
    sheet.write(0,0,"S.N")
    sheet.write(0,1,"Image Dir names")
    sheet.write(0,2,"Individual Train images")
    sheet.write(0,3,"Individual Test images")
    sheet.write(0,4,"True Predicted images")
    sheet.write(0,5,"False Predicted images")
    sheet.write(0,6,"Accuracy")
    
    acc_list = []
        
    for a,img_dict in enumerate(store_list):
        print(a)
        a +=1
        true_p =0
        false_p = 0
        acc = 0
        key0 = list(img_dict.keys())
        value0 = list(img_dict.values())
        
        total_imgs = len(value0[0])
        for img in value0[0]:
            test_emb = extract_feature(img, 
                                        backbone, 
                                        model_root)
            test_emb = test_emb[0].numpy()
            test_emb = normalization(test_emb)
            
            profile_dicts = face_obj.search(np.array([test_emb]))
            
            if(len(profile_dicts[0]) == 0):
                false_p +=1
            else:
                key_1 = list(profile_dicts[0].keys())
                # value0 = list(profile_dicts[0].values())
                
                if(key_1[0] == key0[0]):
                    true_p +=1
                else:
                    false_p +=1
                
                
        acc = (true_p/total_imgs)*100
        acc_list.append(acc)
        
        sheet.write(a,0, a)
        sheet.write(a,1, key0[0])
        sheet.write(a,2, train_img_count)
        sheet.write(a,3, total_imgs)
        sheet.write(a,4, true_p)
        sheet.write(a,5, false_p)
        sheet.write(a,6, acc)
        i = a
        
    total_acc = sum(acc_list)/len(acc_list)
    i += 2
    sheet.write(i,1, "Total Accuracy")
    sheet.write(i,6, total_acc)
    wb.save('excel.xlsx')
    print('-----------------Saved to Excel-------------------')
    
    
                
                