#!/usr/bin/env python3
"""
main.py - get annotation distribution
"""

from scipy.io import loadmat
import numpy as np
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

    data_directory = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Dataset'

    bbox_areas = []

    #extract bbox data for plotting
    for file in os.listdir(data_directory):
        if file.endswith('.mat'):
            process_file(os.path.join(data_directory, file), bbox_areas)

    
    figure, axis = plt.subplots(1, 2) 

    axis[0].hist(bbox_areas, bins=20, edgecolor = "black")
    axis[0].set_title("Counts of Bounding Boxes by Area")
    axis[0].set_xlabel("Bounding Box Area (px)")
    axis[0].set_ylabel("Count")
    axis[0].minorticks_on()


    N, bins, patches = axis[1].hist(np.array(bbox_areas)/1024, bins=20, edgecolor = "black")
    for i, bin in enumerate(bins):
        if bin < .99:
            patches[i].set_facecolor('r')
    axis[1].set_title("Counts of Bounding Boxes by Area After Pooling")
    axis[1].set_xlabel("Bounding Box Area (px)")
    axis[1].set_ylabel("Count")
    axis[1].minorticks_on()
    plt.show()



if __name__ == "__main__":
    """Run from Command Line"""
    main()
