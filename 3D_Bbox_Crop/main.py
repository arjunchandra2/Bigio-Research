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
    im = io.MultiImage(image_path)
    im_frames = im[0]
    pil_frames = []
    #skimage -> PIL for easier cropping and displaying 
    for im in im_frames:
        image = Image.fromarray(im, mode="RGB")
        pil_frames.append(image)
        
    return pil_frames

def crop_bboxes(frames):
    pass

def main():
    """ Main"""

    #OS LOGIC HERE TO READ ALL .MAT FILES IN DIRECTORY
    
    #Read in image and store z_stack in array
    image_path = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/RGB_trans_corrweight_1120.tif'
    im_frames = process_image(image_path)
    
    #Reading in .mat dat and creating bboxes - should be done for each .tif image's corresponding .mat file
    data_path = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/char_annot.mat'
    annotations = load_annotations(data_path)
    add_bboxes(annotations)

    crop_bboxes(im_frames)


    



if __name__ == "__main__":
    """Run from Command Line"""
    main()
