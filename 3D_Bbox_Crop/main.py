#!/usr/bin/env python3
"""
main.py - driver code for cropping annotations
"""
from bbox import Bbox
#from mat4py import loadmat
from scipy.io import loadmat

def main():
    """ Main"""

    #OS LOGIC HERE TO READ ALL .MAT FILES IN DIRECTORY
    data = loadmat('/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/char_annot.mat')

    print(data['annotations'])
    print(data['annotations'][0][2][0].strip()) #strip whitespace from class

    coords_example = data['annotations'][0][5]
    z_plane_example = data['annotations'][0][4]
    test2 = Bbox(coords_example[0][0],coords_example[0][1],coords_example[0][2], \
                coords_example[0][3], z_plane_example[0][0], "swelling")

    print(test2)

    


if __name__ == "__main__":
    """Run from Command Line"""
    main()
