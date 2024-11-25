#!/usr/bin/env python3
"""
build_dataset.py - Crops bounding boxes from images and saves them in a new directory. No labels are saved (unsupervised learning).
"""

import os
import hashlib
import sys
sys.path.insert(0, '/projectnb/npbssmic/ac25')
import utils

#maximum defect size 
MAX_SIZE = 48

def remove_duplicates(image_dir):
    """Remove duplicate .png images from the specified directory."""
    seen_hashes = set()  # To store the hashes of images
    removed_files = 0  # To count the removed files
    
    # Iterate through all files in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            
            # Calculate the hash of the current image
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()  # MD5 hash (can be changed to SHA256)
            
            # If the image hash has already been seen, it's a duplicate
            if image_hash in seen_hashes:
                os.remove(image_path)  # Remove the duplicate image
                removed_files += 1
            else:
                seen_hashes.add(image_hash)  # Mark this image as seen

    # Print the number of removed duplicate files
    print(f"Removed {removed_files} duplicate images.")



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


    remove_duplicates(save_dir + "/train")
    remove_duplicates(save_dir + "/val")
    print("Number of defects saved: ", num_saved)
    print("Number of defects skipped: ", num_skipped)


if __name__ == "__main__":
    """Run from Command Line"""
    main()