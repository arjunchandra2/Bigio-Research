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
CLASS_NUM = {'Defect': 1, 'Swelling': 2, 'Vesicle': 3}

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


def get_overlaps(left, upper, right, lower, z_plane):
    """
    - Returns list of bounding boxes exceeding overlap threshold for given window in given plane
    - Crops BBoxes as necessary
    - Remove Bboxes from tracking
    """
    overlaps = []
    
    for bbox in Bbox.bboxes_unseen[z_plane]:
        #bbox is in the window - check overlap
        if left <= bbox.top_left_x <= right and upper <= bbox.top_left_y <= lower:
            total_area = bbox.width * bbox.height

            if bbox.top_left_x + bbox.width > right:
                width_inside = right-bbox.top_left_x
            else:
                width_inside = bbox.width
            if bbox.top_left_y + bbox.height > lower:
                height_inside = lower-bbox.top_left_y
            else:
                height_inside = bbox.height

            area_inside = width_inside * height_inside
            
            if area_inside/total_area > Bbox.OVERLAP_THRESHOLD:
                #crop bbox and add to overlap set
                bbox.width = width_inside
                bbox.height = height_inside
                overlaps.append(bbox)

    #remove overlaps from tracked bboxes
    for overlap in overlaps:
        Bbox.bboxes_unseen[z_plane].remove(overlap)
        Bbox.count -= 1

    return overlaps

def save_annotations_yolo(left, upper, bboxes, im_path):
    """
    - Save annotations in yolo format .txt file matching image path
    """

    #change extension
    file_path = im_path[:-3] + 'txt'

    f = open(file_path, 'w')

    for bbox in bboxes:
        c_id = CLASS_NUM[bbox.class_name]
        cx = bbox.center_x()
        cy = bbox.center_y()
        width = bbox.width
        height = bbox.height

        #normalize coordinates
        cx_n = (cx-left)/WINDOW_SIZE
        cy_n = (cy-upper)/WINDOW_SIZE
        width_n = width / WINDOW_SIZE
        height_n = height / WINDOW_SIZE

        entry = str(c_id) + ' ' + str(cx_n) + ' ' + str(cy_n) + ' ' + str(width_n) + ' ' + str(height_n) + '\n'
        f.write(entry)
     
    f.close()   

def crop_bboxes(frames, im_path):
    """
    - Crop images around bounding box in each frame as we go and check for overlap
    - Note that z-plane is one-indexed as opposed to frames array
    - Images are saved to working directory with z_plane and crop number
    """

    for i in range(len(frames)):
        if i + 1 in Bbox.bboxes_unseen:
            #crop number
            j = 0
            #while there are boxes left to process in z_plane i+1
            while(Bbox.bboxes_unseen[i+1]):
                #get the last bbox
                current_bbox = Bbox.bboxes_unseen[i+1].pop()
                Bbox.count -= 1
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
                right = left + WINDOW_SIZE
                lower = upper + WINDOW_SIZE

                overlap_bboxes = get_overlaps(left, upper, right, lower, i+1)

                overlap_bboxes.append(current_bbox)

                cropped_im = frames[i].crop((left, upper, right, lower))
                save_path = im_path[:-3] + '(' + str(i+1) + '_' + str(j) + ')' +  '.png'
                print("Saving image......." + save_path)
                cropped_im.save(save_path)
                j += 1

        

                #MODIFY ANNOTATION FORMAT HERE
                save_annotations_yolo(left, upper, overlap_bboxes, save_path)


    
    assert Bbox.count == 0
    
                


def main():
    """ Main"""

    #OS LOGIC HERE TO READ ALL .MAT FILES IN DIRECTORY
    
    #Read in image and store z_stack in array
    image_path = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/RGB_trans_corrweight_1120.tif'
    im_frames = process_image(image_path)

    #Reading in .mat dat and creating bboxes - should be done for each .tif image's corresponding .mat file
    data_path = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Imaging_Scrap1/RGB_trans_corrweight_1120.tif.mat'
    annotations = load_annotations(data_path)
    add_bboxes(annotations)

    crop_bboxes(im_frames, image_path)
    



if __name__ == "__main__":
    """Run from Command Line"""
    main()
