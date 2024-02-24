#!/usr/bin/env python3
"""
main.py - get annotation distribution
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt
import os


def process_file(filename, bbox_areas):
    """
    - Load annotations from .mat file and add bbox areas to bbox_areas
    - Bounding box coordinates may be redundant if a defect exists in multiple
      planes but we will include this anyways since they will all be cropped
      and included seperately in the dataset so our counts should reflect this
      (Augmentation will also scale by the same factor so the distrution is still valid)
    """
    data = loadmat(filename)
    bbox_coords = data['annotations'][0][5]
    for bbox in bbox_coords:
        #add area
        bbox_areas.append(bbox[2]*bbox[3])


def main():
    """ Main"""

    data_directory = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Anna'

    bbox_areas = []

    #extract bbox data for plotting
    for file in os.listdir(data_directory):
        if file.endswith('.mat'):
            process_file(os.path.join(data_directory, file), bbox_areas)
    
    print(len(bbox_areas))
    
    plt.hist(bbox_areas)
    plt.title("Counts of Bounding Boxes by Area")
    plt.xlabel("Boundning Box Area")
    plt.ylabel("Count")
    plt.show()



if __name__ == "__main__":
    """Run from Command Line"""
    main()
