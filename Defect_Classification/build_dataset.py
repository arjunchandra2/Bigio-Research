#!/usr/bin/env python3
"""
build_dataset.py - Crops bounding boxes from images and saves them in a new directory. No labels are saved (unsupervised learning).
"""

import os
import sys
sys.path.insert(0, "..")

import utils

#maximum defect size 
MAX_SIZE = 48


def main():
    dataset_dir = "/projectnb/npbssmic/ac25/RGB_Data_Anna"
    #images to be used for validation
    val_images = ['11_X32342_Y17459.tif']
    save_dir = "/projectnb/npbssmic/ac25/Defect_Classification/unsupervised_dataset"

    if(os.path.exists(save_dir)):
        os.system('rm -fr "%s"' % save_dir)

    #create directory tree 
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir, 'train'))
    os.mkdir(os.path.join(save_dir, 'val'))

    num_skipped = 0
    num_saved = 0
    j = 0

    for file in os.listdir(dataset_dir):
        if file.endswith('.tif'):

            image_path = os.path.join(dataset_dir, file)
            data_path = image_path + '.mat'
            
            #if the file has been annotated then we crop
            if os.path.exists(data_path):
                #Read in image and store z_stack in array of PIL objects
                im_frames = utils.process_image(image_path)
                
                #reading .mat: class_type, z_plane, bbox_coord
                annotations = utils.load_annotations(data_path)
                z_planes = annotations['z_plane']

                if file in val_images:
                    save_path_parent = save_dir + "/val"
                else:
                    save_path_parent = save_dir + "/train"

                for i in range(len(z_planes)):
                    z_plane = z_planes[i]
                    #only crop from planes 8-21
                    if z_plane >=8 and z_plane <= len(im_frames)-4:
                        #get bbox (top left x, top left y, width, height)
                        bbox = annotations['bbox_coord'][i]
                    
                        #crop defect -> padding -> save to new directory
                        if bbox[2] <= MAX_SIZE and bbox[3] <= MAX_SIZE:

                            left = bbox[0]
                            upper = bbox[1] 
                            right = bbox[0] + bbox[2]
                            lower = bbox[1] + bbox[3] 

                            width_adjust = MAX_SIZE - bbox[2]
                            height_adjust = MAX_SIZE - bbox[3]

                            left_adjust = width_adjust//2
                            right_adjust = width_adjust - left_adjust
                            upper_adjust = height_adjust//2
                            lower_adjust = height_adjust - upper_adjust

                            left -= left_adjust
                            right += right_adjust
                            upper -= upper_adjust
                            lower += lower_adjust

                            #defect should still be in bounds after adding border 
                            if left > 0 and upper > 0 and right < im_frames[z_plane-1].size[0] and lower < im_frames[z_plane-1].size[1]:
                                defect = im_frames[z_plane-1].crop((left, upper, right, lower))

                                save_path = os.path.join(save_path_parent, file[:-4] + '_z' + str(z_plane) + '_defect' + str(j) + '.png')
                                print("Saving ", save_path)

                                defect.save(save_path)
                                num_saved += 1
                                j+=1
                                
                            else:
                                num_skipped += 1

                        else:
                            num_skipped += 1


    print("Number of defects saved: ", num_saved)
    print("Number of defects skipped: ", num_skipped)


if __name__ == "__main__":
    """Run from Command Line"""
    main()