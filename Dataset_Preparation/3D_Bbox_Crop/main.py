#!/usr/bin/env python3
"""
main.py - Cropping bounding boxes and creating new directory with images and corresponding annotations in YOLO format
"""

from bbox import Bbox
from scipy.io import loadmat

import torch
from torchvision import transforms
from PIL import Image
from skimage import io

import numpy as np
import random
import os
import time
import sys

sys.path.insert(0, "/projectnb/npbssmic/ac25")
import utils
from Annotation_Filtering.cnn import FilteringCNN

#class encodings -> map all to zero for single class
CLASS_NUM = {'Defect': 0, 'Swelling': 0, 'Vesicle': 1}

#Keep track of number of images created in train and val, num bboxes missed, and num bboxes cleaned by CNN
NUM_VAL = 0
NUM_TRAIN = 0
NUM_MISSED = 0
NUM_CLEANED = 0

#Parameter to set for cropping
WINDOW_SIZE = 100

#Augmentation parameters
AUGMENTATION = False
NUM_CROPS = 2

#model path
CNN_MODEL_PATH = '/projectnb/npbssmic/ac25/Annotation_Filtering/model2.pt'

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

    #for ensuring clean annotations
    Bbox.last_bbox = None


def get_overlaps(left, upper, right, lower, z_plane):
    """
    - Returns list of bounding boxes exceeding overlap threshold for given window in given plane
    - Crops BBoxes as necessary
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

    return overlaps


def save_annotations_yolo(left, upper, bboxes, data_save_path, z, i, theta = 0):
    """
    - Save annotations in yolo format .txt file matching image path
    - optional theta parameter to rotate bounding boxes (used for augmentation)
    """

    #change extension
    file_path = data_save_path[:-3] + '(' + str(z+1) + '_' + str(i) + ')' +  '.txt'

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

        #rotate bounding box if rotation is specified
        if theta != 0:
            #translate (cx_n, cxy_n) to coordinate system with origin at center
            cx_coordinate = cx_n - 0.5
            cy_coordinate = 0.5 - cy_n 
            #counterclockwise rotation by theta
            cx_rotated = cx_coordinate * np.cos(np.deg2rad(theta)) - cy_coordinate * np.sin(np.deg2rad(theta))
            cy_rotated = cx_coordinate * np.sin(np.deg2rad(theta)) + cy_coordinate * np.cos(np.deg2rad(theta))
            #undo coordinate translation 
            cx_n = 0.5 + cx_rotated
            cy_n = 0.5 - cy_rotated

            #swap width and height of bbox if needed
            if theta == 90 or theta == 270:
                temp_width = width_n
                width_n = height_n
                height_n = temp_width


        entry = str(c_id) + ' ' + str(cx_n) + ' ' + str(cy_n) + ' ' + str(width_n) + ' ' + str(height_n) + '\n'
        f.write(entry)
     
    f.close()   


def remove_blurry(frames):
    """
    - Removes bounding boxes from planes 1-7 and 22-25 since they are too blurry for annotations
    and only show up in the data due to bugs in original annotation software 
    - z plane is one-indexed
    """

    for z in range(1,8):
        if z in Bbox.bboxes_unseen:
            Bbox.bboxes_unseen[z] = []
    
    for z in range(len(frames) - 3, len(frames)+1):
        if z in Bbox.bboxes_unseen:
            Bbox.bboxes_unseen[z] = []


def clean_annotations(frames, input_size=48, input_channels=3, threshold=0.5, save_dir="/projectnb/npbssmic/ac25/Dataset_Preparation/results_removed/"):
    """
    - Cleans annotations by removing bounding boxes classified as unclean by pretrained CNN
    - input_size and input_channels are expected input dimensions for CNN
    - CNN output is considered 1 (unclean annotation) if greater than threshold
    """
    global NUM_CLEANED

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    #load model and transforms 
    cnn = FilteringCNN(input_size=input_size, input_channels=input_channels)
    cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location='cpu'))
    cnn.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),  # PIL -> Tensor and [0,255] -> [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # *We are not normalizing to our dataset's actual mean and stdev here*
        ])

    #run all annotations through CNN
    for z in range(len(frames)):
        if z + 1 in Bbox.bboxes_unseen:
            filtered_bboxes = []

            for current_bbox in Bbox.bboxes_unseen[z+1]:
                #get cropping window
                left = current_bbox.top_left_x
                upper = current_bbox.top_left_y
                right = left + current_bbox.width
                lower = upper + current_bbox.height

                #If bbox < input size we add border 
                if current_bbox.width <= input_size and current_bbox.height <= input_size:
                    width_adjust = input_size - current_bbox.width
                    height_adjust = input_size - current_bbox.height

                    left_adjust = width_adjust//2
                    right_adjust = width_adjust - left_adjust
                    upper_adjust = height_adjust//2
                    lower_adjust = height_adjust - upper_adjust

                    left -= left_adjust
                    right += right_adjust
                    upper -= upper_adjust
                    lower += lower_adjust

                    #defect should still be in bounds after adding border 
                    if left > 0 and upper > 0 and right < frames[z].size[0] and lower < frames[z].size[1]:
                        defect = frames[z].crop((left, upper, right, lower))
                        defect = defect.resize((input_size, input_size))
                
                #bbox is greater than input size, we just crop and resize 
                else:
                    defect = frames[z].crop((left, upper, right, lower))
                    defect = defect.resize((input_size, input_size))
                
                #apply transforms and add batch dim
                defect_input = transform(defect)
                defect_input = torch.unsqueeze(defect_input, 0)

                #get CNN prediction: only keep annotation if less than threshold
                if cnn(defect_input).item() < threshold:
                    filtered_bboxes.append(current_bbox)
                else:
                    NUM_CLEANED += 1
                    defect.save(save_dir + "removed_" + str(NUM_CLEANED) + ".png")
            
            #overwrite current z-plane bboxes to only include filtered bboxes
            Bbox.bboxes_unseen[z+1] = filtered_bboxes


def crop_bboxes(frames, im_save_path, data_save_path):
    """
    - Crop images around bounding box in each frame as we go and check for overlap
    - Z-plane is one-indexed as opposed to frames array
    - Images are saved to working directory with z_plane and crop number
    - Remove bounding boxes from first few and last few planes to clean up buggy annotations 
    """
    global NUM_MISSED
    global NUM_VAL
    global NUM_TRAIN

    #remove blurry frames to ensure clean annotations
    remove_blurry(frames)
    #remove annotation bugs with CNN classifier 
    clean_annotations(frames)

    for z in range(len(frames)):
        if z + 1 in Bbox.bboxes_unseen:
            #crop number
            i = 0
            
            #loop over all bboxes in current plane without removal 
            for current_bbox in Bbox.bboxes_unseen[z+1]:
                
                #window size is too small to capture bbox - we just skip (can occur due to buggy annotations)
                if current_bbox.width > WINDOW_SIZE or current_bbox.height > WINDOW_SIZE:
                    NUM_MISSED += 1
                    continue

                #set random cropping bounds - can be used for data augmentation if model architecture is not robust to translation
                #ensuring the window does not exceed the image (this logic might need checking)
                if current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width > 0:
                    leftx = current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width
                else:
                    leftx = 0
                if current_bbox.top_left_x + WINDOW_SIZE > frames[z].size[0]:
                    rightx = current_bbox.top_left_x - (current_bbox.top_left_x + WINDOW_SIZE - frames[z].size[0]) 
                else:
                    rightx = current_bbox.top_left_x

                if current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height > 0:
                    bottomy = current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height
                else:
                    bottomy = 0
                if current_bbox.top_left_y + WINDOW_SIZE > frames[z].size[1]:
                    topy = current_bbox.top_left_y - (current_bbox.top_left_y + WINDOW_SIZE - frames[z].size[1]) 
                else:
                    topy = current_bbox.top_left_y
                
                left = random.randint(leftx, rightx)
                upper = random.randint(bottomy, topy)
                right = left + WINDOW_SIZE
                lower = upper + WINDOW_SIZE

                #get all overlapping bboxes in current window
                overlap_bboxes = get_overlaps(left, upper, right, lower, z+1)

                cropped_im = frames[z].crop((left, upper, right, lower))
                #cropped_im = utils.convert_to_grayscale(cropped_im)
                
                save_path = im_save_path[:-3] + '(' + str(z+1) + '_' + str(i) + ')' +  '.png'
                print("Saving image......." + save_path)
                cropped_im.save(save_path)
    
                #MODIFY ANNOTATION FORMAT HERE
                save_annotations_yolo(left, upper, overlap_bboxes, data_save_path, z, i)

                i += 1
                
                if 'valid' in save_path:
                    NUM_VAL += 1
                else:
                    NUM_TRAIN += 1


    #remove all bounding boxes from Bbox class after the image has been processed
    Bbox.bboxes_unseen = {}
    
    
                
def crop_bboxes_aug(frames, im_save_path, data_save_path):
    """
    - Same as crop_bboxes but with augmentation applied (see info.txt)
    - bboxes are rotated by save_annotations_yolo
    """
    global NUM_MISSED
    global NUM_TRAIN

    #remove blurry frames to ensure clean annotations
    remove_blurry(frames)
    #remove annotation bugs with CNN classifier 
    clean_annotations(frames)

    for z in range(len(frames)):
        if z + 1 in Bbox.bboxes_unseen:
            #crop number
            i = 0

            #loop over all bboxes in current plane without removal 
            for current_bbox in Bbox.bboxes_unseen[z+1]:

                #window size is too small to capture bbox - we just skip (can occur due to buggy annotations)
                if current_bbox.width > WINDOW_SIZE or current_bbox.height > WINDOW_SIZE:
                    NUM_MISSED += 1
                    continue

                #set random cropping bounds - can be used for data augmentation if model architecture is not robust to translation
                #ensuring the window does not exceed the image (this logic might need checking)
                if current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width > 0:
                    leftx = current_bbox.top_left_x - WINDOW_SIZE + current_bbox.width
                else:
                    leftx = 0
                if current_bbox.top_left_x + WINDOW_SIZE > frames[z].size[0]:
                    rightx = current_bbox.top_left_x - (current_bbox.top_left_x + WINDOW_SIZE - frames[z].size[0]) 
                else:
                    rightx = current_bbox.top_left_x

                if current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height > 0:
                    bottomy = current_bbox.top_left_y - WINDOW_SIZE + current_bbox.height
                else:
                    bottomy = 0
                if current_bbox.top_left_y + WINDOW_SIZE > frames[z].size[1]:
                    topy = current_bbox.top_left_y - (current_bbox.top_left_y + WINDOW_SIZE - frames[z].size[1]) 
                else:
                    topy = current_bbox.top_left_y


                #AUGMENTATION:     
                #multiple random croppings around bounding box
                for n in range(NUM_CROPS):
                    #choose random window bounds
                    left = random.randint(leftx, rightx)
                    upper = random.randint(bottomy, topy)
                    right = left + WINDOW_SIZE
                    lower = upper + WINDOW_SIZE

                    overlap_bboxes = get_overlaps(left, upper, right, lower, z+1)

                    cropped_im = frames[z].crop((left, upper, right, lower))
                    #cropped_im = utils.convert_to_grayscale(cropped_im)
                    

                    #regular and transformed/swapped channels
                    for channels in ["original", "swapped"]:
                        if channels == "swapped":
                            cropped_im = utils.swap_channels(cropped_im)
                        
                        #4 orientations
                        for theta in [0, 90, 180, 270]:
                            cropped_im_rotated = cropped_im.rotate(theta)

                            save_path = im_save_path[:-3] + '(' + str(z+1) + '_' + str(i) + ')' +  '.png'
                            print("Saving image......." + save_path)
                            cropped_im_rotated.save(save_path)
        
                            #MODIFY ANNOTATION FORMAT HERE
                            save_annotations_yolo(left, upper, overlap_bboxes, data_save_path, z, i, theta)

                            i += 1
                            NUM_TRAIN += 1

   
    #remove all bounding boxes from Bbox class after the image has been processed
    Bbox.bboxes_unseen = {}
                


def main():
    """ Main"""

    #READ AND PROCESS ALL .MAT FILES IN DIRECTORY AND SAVE RESULTS TO ./RESULTS
    data_directory = '/projectnb/npbssmic/ac25/RGB_Data_Anna'
    
    #images to be used for validation
    val_images = ['11_X32342_Y17459.tif']

    #timing
    start_time = time.perf_counter()

    results_dir = "/projectnb/npbssmic/ac25/Dataset_Preparation/results"
    
    if(os.path.exists(results_dir)):
        os.system('rm -fr "%s"' % results_dir)

    
    #create directory tree 
    os.mkdir(results_dir)
    os.mkdir(os.path.join(results_dir, 'train'))
    os.mkdir(os.path.join(results_dir, 'train', 'images'))
    os.mkdir(os.path.join(results_dir, 'train', 'labels'))
    os.mkdir(os.path.join(results_dir, 'valid'))
    os.mkdir(os.path.join(results_dir, 'valid', 'images'))
    os.mkdir(os.path.join(results_dir, 'valid', 'labels'))


    for file in os.listdir(data_directory):
        if file.endswith('.tif'):

            image_path = os.path.join(data_directory, file)
            data_path = image_path + '.mat'
            
            #if the file has been annotated then we crop
            if os.path.exists(data_path):
                #Read in image and store z_stack in array of PIL objects
                im_frames = utils.process_image(image_path)
                
                #reading .mat and adding bboxes to Bbox class
                annotations = utils.load_annotations(data_path)
                add_bboxes(annotations)

                if file in val_images:
                    image_save_path = os.path.join(results_dir, 'valid', 'images', file)
                    data_save_path = os.path.join(results_dir, 'valid', 'labels', file)

                    #no augmentation for validation images
                    crop_bboxes(im_frames, image_save_path, data_save_path)
                else:
                    image_save_path = os.path.join(results_dir, 'train', 'images', file)
                    data_save_path = os.path.join(results_dir, 'train', 'labels', file)

                    #crop and save bboxes using PIL img array 'frames' and Bbox class 
                    if AUGMENTATION:
                        crop_bboxes_aug(im_frames, image_save_path, data_save_path)
                    else:
                        crop_bboxes(im_frames, image_save_path, data_save_path)
                

    finish_time = time.perf_counter()

    print()
    print("Succesfully created dataset in ~", (finish_time-start_time)//60, "minutes.")
    print("Dataset size:")
    print("Training set", NUM_TRAIN , "images")
    print("Validation set", NUM_VAL , "images")
    print("A total of", NUM_MISSED, "bounding boxes could not be cropped with a window size of", WINDOW_SIZE)
    print("A total of", Bbox.BBOXES_REMOVED, "bounding boxes were removed as duplicates/jumping multiple planes")
    print("A total of", NUM_CLEANED, "bounding boxes were removed as unclean annotations by the CNN")



if __name__ == "__main__":
    """Run from Command Line"""
    main()




