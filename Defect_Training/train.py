# -*- coding: utf-8 -*-
"""
Training script for defect detection
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

#import os 
#from os import listdir
#import torch
#import torchvision
#from IPython import display
#from IPython.display import display, Image



#Check device before training:
#import os
#print(os.getenv("CUDA_VISIBLE_DEVICES"))


from ultralytics import YOLO

d_path = "/projectnb/npbssmic/ac25/Defect_Training/data.yaml"

model = YOLO("yolov8n.pt")

results = model.train(data = d_path,
                      epochs = 100,
                      patience = 20,
                      device = [1],
                      imgsz = 640, 
                      save = True,
                      plots = True,
                      lr0= 0.0101,
                      lrf= 0.01055,
                      momentum= 0.94049,
                      weight_decay= 0.00044,
                      warmup_epochs= 2.95485,
                      warmup_momentum= 0.81488,
                      box= 7.60451,
                      cls= 0.44261,
                      dfl= 1.58377,
                      hsv_h= 0.01359,
                      hsv_s= 0.55782,
                      hsv_v= 0.39803,
                      degrees= 0.0,
                      translate= 0.1118,
                      scale= 0.43128,
                      shear= 0.0,
                      perspective= 0.0,
                      flipud= 0.0,
                      fliplr= 0.52765,
                      mosaic= 1.0,
                      mixup= 0.0,
                      copy_paste= 0.0,
                      optimizer='AdamW',
                      single_cls=True
                      )