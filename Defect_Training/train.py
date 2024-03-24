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

import os 
from os import listdir
import torch
import torchvision
from IPython import display
from IPython.display import display, Image
from ultralytics import YOLO

d_path = "/projectnb/npbssmic/ac25/Defect_Training/data.yaml"

model = YOLO("yolov8n.pt")

results = model.train(data = d_path, 
                      epochs = 100,
                      imgsz = 640, 
                      save = True,
                      plots = True,
                      hsv_h = 0,
                      hsv_s = 0,
                      hsv_v = 0,
                      translate = 0,
                      scale = 0,
                      fliplr = 0,
                      mosaic = 0, 
                      )