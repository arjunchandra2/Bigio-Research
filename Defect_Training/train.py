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

results = model.train(
    mode = 'train',
    model = 'yolov8n.pt',
    data = d_path,
    epochs = 40,
    patience = 50,
    batch = 16,
    imgsz = 640,
    save = False,
    save_period = -1,
    cache = False,
    device = [3],
    workers = 8,
    project = None,
    name = None,
    exist_ok = False,
    pretrained = True,
    optimizer= 'AdamW',
    verbose = True,
    seed = 0,
    deterministic = True,
    single_cls = False,
    rect = False,
    cos_lr = False,
    close_mosaic = 10,
    resume = False,
    amp = True,
    fraction = 1.0,
    profile = False,
    freeze = None,
    overlap_mask = True,
    mask_ratio = 4,
    dropout = 0.0,
    val = True,
    split= 'val',
    save_json = False,
    save_hybrid = False,
    conf = None,
    iou = 0.7,
    max_det = 300,
    half = False,
    dnn = False,
    plots = True,
    source = None,
    show = False,
    save_txt = False,
    save_conf = False,
    save_crop = False,
    show_labels = True,
    show_conf = True,
    vid_stride = 1,
    stream_buffer = False,
    line_width = None,
    visualize = False,
    augment = False,
    agnostic_nms = False,
    classes = None,
    retina_masks = False,
    boxes= True,
    format = 'torchscript',
    keras = False,
    optimize = False,
    int8 = False,
    dynamic = False,
    simplify = False,
    opset = None,
    workspace= 4,
    nms = False,
    lr0 = 0.0101,
    lrf = 0.01055,
    momentum = 0.94049,
    weight_decay = 0.00044,
    warmup_epochs = 2.95485,
    warmup_momentum = 0.81488,
    warmup_bias_lr = 0.1,
    box = 7.60451,
    cls = 0.44261,
    dfl = 1.58377,
    pose = 12.0,
    kobj = 1.0,
    label_smoothing = 0.0,
    nbs = 64,
    hsv_h = 0.01359,
    hsv_s = 0.55782,
    hsv_v = 0.39803,
    degrees = 0.0,
    translate = 0.1118,
    scale = 0.43128,
    shear = 0.0,
    perspective = 0.0,
    flipud = 0.0,
    fliplr = 0.52765,
    mosaic = 1.0,
    mixup = 0.0,
    copy_paste = 0.0,
    cfg = None,
    tracker = 'botsort.yaml',
    save_dir = 'runs/detect/train100'
                      )

