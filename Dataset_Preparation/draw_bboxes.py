#!/usr/bin/env python3
"""
draw_bboxes.py - Draw bbox on image for visualization (reads .txt in Yolo format)
"""
from PIL import Image
from PIL import ImageDraw


#Defect is red, Swelling is green, Vesicle is blue
CLASS_COLORS = {0: "red", 1 : "green", 2 : "blue"}



def main():
    """ Main"""

    #Add OS logic here if needed to loop over directory, draw bboxes, and save images in new directory 
    image_path = "/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Dataset/results/images/11_X32342_Y17459.(8_221).png"
    annotation_path = "/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Dataset/results/annotations/11_X32342_Y17459.(8_221).txt"
    save_path = ""


    im = Image.open(image_path)
    draw = ImageDraw.Draw(im)

    #assuming square image
    im_size = im.size[0]

    annotations = open(annotation_path, 'r')

    bboxes = annotations.readlines()

    #draw each bounding box on the image
    for bbox in bboxes:
        bbox = bbox.split()
        bbox = [float(x) for x in bbox]
        #bbox is now [class, centerx, centery, width, height] <- normalized
        color = CLASS_COLORS[bbox[0]]
        centerx = bbox[1] * im_size
        centery = bbox[2] * im_size
        width = bbox[3] * im_size
        height = bbox[4] * im_size

        tlx = centerx - width/2
        tly = centery - height/2
        blx = tlx + width
        bly = tly + height
        draw.rectangle([tlx,tly,blx,bly], outline=color)

    im.show()

    #save the image anywhere you like, use a different filepath than the original to not overwrite
    if save_path:
        im.save(save_path)



        



if __name__ == "__main__":
    """Run from Command Line"""
    main()
