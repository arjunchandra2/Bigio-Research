#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:53:43 2023

@author: ac25 (Arjun Chandra)

This program models the blur propogation of 3-D Brain Scans so that we can train on open source datasets
and implement transfer learning since MRI scan data is limited and expensive and lends itself to overfitting.
There are degrees of randomness that are implemented in this version, but perhaps a better implementation will
consider the bounding box when propogating the blur. 
"""
from PIL import Image, ImageFilter
import numpy as np
import math
import random
import os
import shutil

mod_size = 600
grid_dim = 12
num_channels = 23
num_prop = 3


def b_scheme_rad(cx, cy, x, y):
    """
    returns kernel size for the cell at (x,y) when current center
    cell is at (cx,cy) - radial implementation
    """
    return (math.sqrt((x-cx)**2 + (y-cy)**2))/math.pi + 1
    #**HARD CODED PI**

def b_scheme_guass(cx,cy,x,y):
    """
    returns kernel size for the cell at (x,y) when current center
    cell is at (cx,cy) - guassian implementation
    """
    pass


def shift_center(cx, cy, direction):
    """
    shifts center (cx,cy) in cardinal direction for next frame 
    """
    if direction == "ul":
        return (cx-1, cy+1)
    elif direction == "u":
        return (cx, cy+1)
    elif direction == "ur":
        return (cx+1, cy+1)
    elif direction == "l":
        return (cx-1, cy)
    elif direction == "r":
        return (cx+1, cy)
    elif direction == "dl":
        return (cx-1, cy-1)
    elif direction == "d":
        return (cx, cy-1)
    elif direction == "dr":
        return (cx+1, cy-1)
    else:
        raise ValueError("Invalid direction")

        
        
def save_channels(frames):
    """
    Saves channels to a directory named test 1 for inspection
    """
    test_dir = "/projectnb/npbssmic/ac25/g_blur/test1"
    
    
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        
    os.mkdir(test_dir)
  
    for i in range(len(frames)):
        frames[i].save(f"{test_dir}/frame{i}.png")
        


def write_info(cx, cy, f_dir, b_dir, center_frame):
    """
    Writes run data to file - for testing/debugging
    """

    f = open("/projectnb/npbssmic/ac25/g_blur/test1/info.txt", "a+")
    
    f.write("//RUN INFO//\n\n")
    f.write(f"mod_size = {mod_size}\ngrid_dim = {grid_dim}\n\
num_channels = {num_channels}\nnum_prop = {num_prop}\n\nRandomness:\n\
center_frame = {center_frame}\ncenter = {cx},{cy}\nforward_direction = \
{f_dir}\nbackward_direction = {b_dir}")
               
    f.close()



def main():
    im = Image.open('/projectnb/npbssmic/ac25/g_blur/cat2.jpg')

    #resize image and convert grayscale    
    im = im.resize((mod_size,mod_size))
    im = im.convert("L")

    grid_size = mod_size/grid_dim
    
    #to be populated with channels
    frames = np.array([None]*num_channels)

    #define grid bounds for each cell
    grid_boxes= np.array([[(x*grid_size, y*grid_size, (x+1)*grid_size, 
                           (y+1)*grid_size) for y in range(grid_dim)]
                          for x in range(grid_dim)])

    #create cells
    grid = np.array([[im.crop(grid_boxes[x][y]) for y in range(grid_dim)] 
                      for x in range(grid_dim)])

   
    #GENERATE RANDOMNESS
    
    #generate random center (cannot be on edge) - keep 3 (b_dir, f_dir, permanent)
    cx1=cx2=cx3= np.random.randint(num_prop, grid_dim-num_prop+1)
    cy1=cy2=cy3=np.random.randint(num_prop, grid_dim-num_prop+1)
    
    #generate random cardinal direction for propogation
    direction = random.choice([('ul', 'dr'), ('ur', 'dl'), 
                                  ('l', 'r'), ('u', 'd')])
    p_dir = np.random.randint(0,2)
    f_dir = direction[p_dir]
    b_dir = direction[np.heaviside(1-p_dir, 0).astype(int)] #1-p_dir

    #generate random frame to propogate around    
    center_frame = np.random.randint((num_channels // 2)-num_prop, 
                                     (num_channels // 2) + num_prop +  1)
    
    
    #populate frames before prop
    for i in range(center_frame-num_prop):
        frames[i] = im.filter(ImageFilter.GaussianBlur(
            radius = ((center_frame-num_prop)//(i+1))+2)) #**HARD CODED**
        
    
    #temporary frame so we don't change the original
    temp_frame = im
    
    #pop center frame
    for x in range(grid_dim):
        for y in range(grid_dim):
            im_c = Image.fromarray(grid[x][y], "L")
            blur_image = im_c.filter(ImageFilter.GaussianBlur(
                radius=b_scheme_rad(cx3, cy3, x, y)))
            temp_frame.paste(blur_image, grid_boxes[x][y].astype(int)) 
    
    frames[center_frame] = temp_frame
    
    
    #propogate backwards
    for i in range(center_frame-1, center_frame-num_prop-1, -1):
        cx1, cy1 = shift_center(cx1, cy1, b_dir)
        temp_frame = im
        
        for x in range(grid_dim):
            for y in range(grid_dim):
                im_c = Image.fromarray(grid[x][y], "L")
                blur_image = im_c.filter(ImageFilter.GaussianBlur(
                    radius=b_scheme_rad(cx1, cy1, x, y)))
                temp_frame.paste(blur_image, grid_boxes[x][y].astype(int)) 
                
        frames[i] = temp_frame
        
    #propogate forwards
    for i in range(center_frame+1, center_frame+num_prop+1):
        cx2, cy2 = shift_center(cx2, cy2, f_dir)
        temp_frame = im
        
        for x in range(grid_dim):
            for y in range(grid_dim):
                im_c = Image.fromarray(grid[x][y], "L")
                blur_image = im_c.filter(ImageFilter.GaussianBlur(
                    radius=b_scheme_rad(cx1, cy1, x, y)))
                temp_frame.paste(blur_image, grid_boxes[x][y].astype(int)) 
                
        frames[i] = temp_frame

    #fill in rest of frames
    for i in range(center_frame+num_prop+1, num_channels):
        frames[i] = im.filter(ImageFilter.GaussianBlur(
            radius = (i//(center_frame+num_prop+1))+2)) #**HARD CODED 3**
    
    
    save_channels(frames)
    write_info(cx3, cy3, f_dir, b_dir, center_frame)
        
        
if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()







"""
//OLD TEST CODE//

#create image grid 
grid = np.array([[im.crop((x*grid_size, y*grid_size, (x+1)*grid_size, 
                (y+1)*grid_size)) for x in range(grid_dim)]
                 for y in range(grid_dim)]) 


crop_img = grid[0][0]
# Use GaussianBlur directly to blur the image
blur_image = crop_img.filter(ImageFilter.GaussianBlur(radius=5))
im.paste(blur_image, box)

RBG values condensed after convert("L") to one value from 0 to 255
imdata = list(im.getdata())
print(imdata)

#im.show()

black_box = np.zeros((50,50))
bb = Image.fromarray(black_box)

im.paste(bb, (100,0,150,50))

im.show()
"""
