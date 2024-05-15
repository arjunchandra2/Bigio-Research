# -*- coding: utf-8 -*-
"""
Implenting Yolov8n model for obejct detection on blood cell dataset 
Author: Arjun Chandra 

@software{yolov8_ultralytics,
  author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title = {Ultralytics YOLOv8},
  version = {8.0.0},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
"""

import os 
from os import listdir
import torch
import torchvision
from IPython import display
from IPython.display import display, Image
from ultralytics import YOLO

d_path = "/projectnb/npbssmic/ac25/blood_cell_d/Data/Blood Cell Detection.v3i.yolov8/data.yaml"
s_path = "/projectnb/npbssmic/ac25/blood_cell_d/Data/Blood Cell Detection.v3i.yolov8/test/images"

model = YOLO("yolov8n.yaml")

results = model.train(data= d_path, 
                      epochs = 100)

folder_dir = s_path
for images in os.listdir(folder_dir):
   
    if (images.endswith(".jpg")):
        display(Image(filename=folder_dir + "/" + images, height = 600))

  

pred = model.predict(s_path)




#val= model.detect(data ="/projectnb/npbssmic/ac25/Blood Cell Detection/Data/Blood Cell Detection.v3i.yolov8/test/images")
#model(image)

#loggers='tensorboard', logdir='/projectnb/npbssmic/ac25/Blood Cell Detection/Data/'
#print(torch.__version__)
#print(torchvision.__version__)
