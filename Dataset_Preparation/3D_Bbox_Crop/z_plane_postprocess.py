#!/usr/bin/env python3
"""
z_plane_postprocess.py - Remove bounding boxes which appear in consecutive planes as they are likely transverse axons 
"""

from bbox import Bbox
from main import add_bboxes
from scipy.io import loadmat
from scipy.io import savemat

import sys
sys.path.insert(0, "/projectnb/npbssmic/ac25")
import utils

#IOU threshold to remove bboxes 
IOU_THRESHOLD = 0.5
#Max number of planes a defect can be in 
MAX_PLANES = 2

#EXTRA INFO NEEDED
#color encodings for saving new mat file 
COLOR_ENCODING = {'Defect': [1,0,0], 'Swelling': [0,1,0], 'Vesicle': [0,0,1]}


def load_annotations_full(file_path):
    """
    - Same as utils function but includes confidence values added by model
    """
    annotations = utils.load_annotations(file_path)
    data = loadmat(file_path)
    annotations['conf'] = data['annotations'][0][6]

    return annotations

def add_bboxes(annotations):
    """
    - Add bboxes to Bbox class for tracking
    """
    for i in range(len(annotations['class_type'])):
        class_name = annotations['class_type'][i]
        z_plane = annotations['z_plane'][i]
        coords = annotations['bbox_coord'][i]
        conf = annotations['conf'][i]
        
        bbox = Bbox(coords[0], coords[1], coords[2], coords[3], z_plane, class_name, conf)

    #for ensuring clean annotations
    Bbox.last_bbox = None

def save_cleaned_annotations(clean_annotations, im_path, save_path):
    """
    - Save the cleaned .mat file 
    """
    mat_annotations = {"annotations": [im_path, "YOLOv8", [], [], [], [], []]}

    for z in clean_annotations:
        bboxes = clean_annotations[z]
        for bbox in bboxes:
            mat_annotations['annotations'][2].append(bbox.class_name)
            mat_annotations['annotations'][3].append(list(map(float, COLOR_ENCODING[bbox.class_name])))
            mat_annotations['annotations'][4].append([bbox.z_plane])
            mat_annotations['annotations'][5].append(list(bbox.get_coords()))
            mat_annotations['annotations'][6].append([bbox.conf])
    
    savemat(save_path, mat_annotations)


def main():
    """ Main"""

    #Add OS logic here if needed to loop over directory
    mat_file = "/projectnb/npbssmic/ac25/RGB_Data_Fall_24(9_23)/AD8790_4p1_X15652_Y5312.tif.mat"
    im_path = "/projectnb/npbssmic/ac25/RGB_Data_Fall_24(9_23)/AD8790_4p1_X15652_Y5312.tif"
    save_path = "/projectnb/npbssmic/ac25/RGB_Data_Fall_24(9_23)/AD8790_4p1_X15652_Y5312_postprocessed.tif.mat"
    
    
    #reading .mat and adding bboxes to Bbox class
    annotations = load_annotations_full(mat_file)
    add_bboxes(annotations)
    
    num_planes = max(annotations['z_plane'])
    num_removed = 0 

    #We will remove bounding boxes from here as we go 
    cleaned_bboxes = Bbox.bboxes_unseen

    #iterate over all bounding boxes in order by z_plane 
    for z in range(1, num_planes):
        if z in Bbox.bboxes_unseen:
            for current_bbox in Bbox.bboxes_unseen[z]:
                #we will measure IOU against this bbox
                base_bbox = current_bbox
                overlap_bboxes = []
                consecutive_planes = 1

                for z_next in range(z+1, num_planes):
                    #If there are any bboxes in the next plane 
                    if z_next in Bbox.bboxes_unseen:
                        #Look for any bboxes that overlap > threshold with base bbox
                        #If there are multiple we will remove them all
                        seen = False
                        for candidate_bbox in Bbox.bboxes_unseen[z_next]:
                            iou = utils.compute_iou(base_bbox.get_coords(), candidate_bbox.get_coords())
                            if iou > IOU_THRESHOLD:
                                overlap_bboxes.append(candidate_bbox)
                                if not seen:
                                    consecutive_planes += 1
                                    seen = True
                    #Stop looping if there are no bboxes in the next z_plane 
                    else:
                        break
                    
                    #If we did not find any bboxes which overlap > threshold with base bbox we stop 
                    if not seen:
                        break
                
                #Remove bboxes if they appear in more than MAX_PLANES if we haven't already removed them
                if consecutive_planes > MAX_PLANES:
                    if current_bbox in cleaned_bboxes[z]:
                        cleaned_bboxes[z].remove(current_bbox)
                        num_removed += 1
                    for bbox in overlap_bboxes:
                        if bbox in cleaned_bboxes[bbox.z_plane]:
                            cleaned_bboxes[bbox.z_plane].remove(bbox)
                            num_removed += 1
    
    save_cleaned_annotations(cleaned_bboxes, im_path, save_path)
    print(f"Removed a total of {num_removed} bboxes")

                    


if __name__ == "__main__":
    """Run from Command Line"""
    main()
