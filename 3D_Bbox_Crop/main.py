#!/usr/bin/env python3
"""
main.py - driver code for cropping annotations
"""
from bbox import Bbox
#from mat4py import loadmat
from scipy.io import loadmat


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
    


if __name__ == "__main__":
    """Run from Command Line"""
    main()
