#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:00:44 2023

@author: ac25

Image Pre-processing: radial blur with BBox

Currently implemented with dataset from https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile 
import json 
import diplib as dip

def get_annotations(annotations, im_id):
    """
    Search through the annotations if needed. This will be O(n^2) so it will 
    be more efficient to sort in place if needed O(nlogn).
    
     **Not needed for drone dataset since annotation id's are ordered by image id.
    """

    for i in range(len(annotations['annotations'])):
        if annotations['annotations'][i]['image_id'] == im_id:
            return annotations['annotations'][i]
    
    return None

def main():
    
    annotations_path = "/projectnb/npbssmic/ac25/rad_blur/annotations/merged_train.json"
    base_img_path = '/projectnb/npbssmic/ac25/rad_blur/images/'
    
    with open(annotations_path) as annotations_file:
        annotations = json.load(annotations_file)
        
    
    #Cut off images without annotations
    annotations['images'] = annotations['images'][363:]
    
    #current image to blur
    i = 0
        
    print("Example image entry (COCO Format):", annotations['images'][i])
    
    imgpath_1 = base_img_path + annotations['images'][i]['file_name']
    img1_id = annotations['images'][i]['id']
    
    img1 = Image.open(imgpath_1)
    #img1.show()
    
    print('Example annotation (COCO Format)', annotations['annotations'][i])
    
    img = dip.ImageRead(imgpath_1)

    #Adapative kernel size    
    scale = dip.CreateRadiusCoordinate(img.Sizes()) / 100  #**HARD CODED 
    
    #Get Bounding Box Coordinates
    leftx = int(annotations['annotations'][i]['bbox'][0])
    topy = int(annotations['annotations'][i]['bbox'][1])
    rightx = int(leftx + annotations['annotations'][i]['bbox'][2])
    bottomy = int(topy + annotations['annotations'][i]['bbox'][3])
    
    #set kernel size to 1 for bbox
    scale[leftx:rightx, topy:bottomy] = 1
    
    plt.imshow(scale, cmap = 'gray')
    plt.title("Kernel Size as Distance from Origin")
   
    #Adaptive kernel orientation 
    angle = dip.CreatePhiCoordinate(img.Sizes())
    
    #Apply adapative gauss. Varying kernel size and orientation to create radial blur
    out = dip.AdaptiveGauss(img, [angle, scale], [1,5])
    
    out.Show()
    


if __name__ == "__main__":
    main()
