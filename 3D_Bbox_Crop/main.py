#!/usr/bin/env python3
"""
main.py - driver code for cropping annotations
"""
from bbox import Bbox
#from mat4py import loadmat
from scipy.io import loadmat


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


def main():
    """ Main"""

    #OS LOGIC HERE TO READ ALL .MAT FILES IN DIRECTORY
    annotations = load_annotations('/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/char_annot.mat')
    add_bboxes(annotations)

    



if __name__ == "__main__":
    """Run from Command Line"""
    main()
