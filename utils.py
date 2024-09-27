#!/usr/bin/env python3
"""
utils.py
"""

from scipy.io import loadmat
from PIL import Image
from skimage import io
import numpy as np


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
    - *Can improve efficiency if needed by only storing frames for which annotations are present*
    """
    im = io.MultiImage(image_path)
    im_frames = im[0]
    pil_frames = []
    #skimage -> PIL for easier cropping and displaying 
    for im in im_frames:
        image = Image.fromarray(im, mode="RGB")
        pil_frames.append(image)
        
    return pil_frames


def swap_channels(image):
    """
    - Applies the following transformation to image channels and returns new image
    (R,G,B) -> (R-1.4B, G, B)
    - 4.5s for 10k images
    """
    channels = np.array(image)
    #apply transformation
    channels[:,:,0] = abs(channels[:,:,0] - 1.4*channels[:,:,2])
    #keep values in valid range 
    channels = channels.clip(0,255)

    return Image.fromarray(channels)


def swap_channels_lib(image):
    """
    - Same as swap_channels but using PIL library calls 
    - 6.5s for 10k images
    """
    r, g, b = image.split()
    r = abs(np.asarray(r)-1.4*np.asarray(b))
    r = Image.fromarray(r.clip(0, 255).astype(np.uint8))
    
    return Image.merge("RGB", (r,g,b))


def convert_to_grayscale(image):
    """
    - Converts the RGB image to grayscale by copying the G channel to R and B
    (R,G,B) -> (G, G, G)
    """
    channels = np.array(image)
    #apply transformation
    channels[:,:,0] = channels[:,:,1]
    channels[:,:,2] = channels[:,:,1]

    return Image.fromarray(channels)