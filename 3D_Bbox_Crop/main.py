#!/usr/bin/env python3
"""
main.py - driver code for cropping annotations
"""
from bbox import Bbox
#from mat4py import loadmat
from scipy.io import loadmat
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt
import random

WINDOW_SIZE = 600

def add_bboxes(annotations):
    """
    - Create bbox objects for all of the bounding boxes in each (3D) image
    - All of the bboxes for the image will be stored within the class
    """
    for i in range(len(annotations['class_type'])):
        class_name = annotations['class_type'][i]
        z_plane = annotations['z_plane'][i]
        coords = annotations['bbox_coord'][i]
        
        bbox = Bbox(coords[0], coords[1], coords[2], coords[3], z_plane, class_name)


def load_annotations(file_path):
    """
    - Read in and format annotations from .mat file into dictionary 
    - Keep numpy datatypes
    """

    data = loadmat(file_path)
    class_type = data['annotations'][0][2]
    #strip whitespace
    for i in range(len(class_type)):
        class_type[i] = class_type[i].strip()
    
    z_plane = data['annotations'][0][4]
    z_plane = z_plane.flatten()

    bbox_coord = data['annotations'][0][5]
    
    assert len(class_type) == len(z_plane) == len(bbox_coord)

    annotations_dict = {'class_type': class_type, 'z_plane': z_plane, 'bbox_coord': bbox_coord}
    
    return annotations_dict

def process_image(image_path):
    """
    - Returns list of Pillow images for each frame
    - **Can improve efficiency if needed by only storing frames for which annotations are present
    """
    im = io.MultiImage(image_path)
    im_frames = im[0]
    pil_frames = []
    #skimage -> PIL for easier cropping and displaying 
    for im in im_frames:
        image = Image.fromarray(im, mode="RGB")
        pil_frames.append(image)
        
    return pil_frames

def crop_bboxes(frames):
    """
    - Crop images around bounding box in each frame as we go and check for overlap
    - Note that z-plane is one-indexed 
    """

    for i in range(len(frames)):
        if i + 1 in Bbox.bboxes_unseen:
            #while there are boxes left to process in z_plane i+1
            while(Bbox.bboxes_unseen[i+1]):
                #get the last bbox
                current_bbox = Bbox.bboxes_unseen[i+1].pop()
                #set random cropping bounds - can be used for data augmentation if model architecture is not robust to translation
                #ensuring the window does not exceed the image (this logic might need checking)
                if current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width > 0:
                    leftx = current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width
                else:
                    leftx = 0
                if current_bbox.top_left_x + WINDOW_SIZE > frames[i].size[0]:
                    rightx = current_bbox.top_left_x - (current_bbox.top_left_x + WINDOW_SIZE - frames[i].size[0]) 
                else:
                    rightx = current_bbox.top_left_x

                if current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height > 0:
                    bottomy = current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height
                else:
                    bottomy = 0
                if current_bbox.top_left_y + WINDOW_SIZE > frames[i].size[1]:
                    topy = current_bbox.top_left_y - (current_bbox.top_left_y + WINDOW_SIZE - frames[i].size[1]) 
                else:
                    topy = current_bbox.top_left_y
                
                left = random.randint(leftx, rightx)
                upper = random.randint(bottomy, topy)

                





def main():
    """ Main"""

    #OS LOGIC HERE TO READ ALL .MAT FILES IN DIRECTORY
    
    #Read in image and store z_stack in array
    image_path = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/RGB_trans_corrweight_1120.tif'
    im_frames = process_image(image_path)
    
    print(im_frames[0].size[0])

    #Reading in .mat dat and creating bboxes - should be done for each .tif image's corresponding .mat file
    data_path = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/RGB_trans_corrweight_1120.tif.mat'
    annotations = load_annotations(data_path)
    add_bboxes(annotations)

    crop_bboxes(im_frames)


    



if __name__ == "__main__":
    """Run from Command Line"""
    main()
