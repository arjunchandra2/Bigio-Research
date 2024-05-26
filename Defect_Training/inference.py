"""
Running model inference through Roboflow API/local inference
and stitching predictions for .tif image into .mat format 
"""

from roboflow import Roboflow
from dotenv import load_dotenv
from scipy.io import savemat
from ultralytics import YOLO
import numpy as np
import time
import os


#path to YOLO model 
MODEL_PATH = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Bigio-Research/Defect_Training/best.pt'
#confidence threshold for detections
CONFIDENCE = 0.2
#window size
WINDOW_SIZE = 300
#window overlap
WINDOW_OVERLAP = 0.3

def configure():
    """
    Wrapper for loading environment 
    """
    load_dotenv()



def get_roboflow_pred(im_path):
    """
    - Get model predictions via Roboflow API
    - Using YOLO-NAS model: 0.425 mAP
    """
    rf = Roboflow(api_key=os.getenv('api_key'))
    #print(rf.workspace().projects())
    project = rf.workspace().project("defect-training-5-3")
    model = project.version(4).model
    print(model)

    # infer on a local image - overlap set to 70%
    print(model.predict(im_path, confidence=40, overlap=70).json())

    # visualize your prediction
    #model.predict(im_path, confidence=40, overlap=30).save("/Users/arjunchandra/Desktop/prediction_test.jpg")
    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
    

def get_local_pred(model, image):
    """
    - Image inference from local Yolov8 model 
    - Image can be any of YOLO accepted formats but will switch to only PIL: 
    https://docs.ultralytics.com/modes/predict/#inference-sources
    - Bottleneck for time efficiency: ~25s for 100 subimages w/o batch processing
    - Can speed up with batch processing and/or changing device to gpu
    """ 
    #imgsz = (width, height), recommended to resize to (640,640) -> seems to work fine even for rectangular images
    #resizing maintains aspect ratio using rescale and pad and maintains multiple of 32 (network stride)
    results = model.predict(source=image, conf=CONFIDENCE, imgsz=640, iou=0.7, device='cpu')

    #only a single result if no batch inference
    for result in results:
        print(result.boxes)
        bboxes = result.boxes.numpy().xywh
        
    



def get_inference(im_path):
    """
    Runs model inference on .tif image and returns formatted .mat file
    for viewing model annotations in Matlab software
    """
    pass

    #read in image and process plane by plane 

    #for each plane, split into subimages via sliding window with overlap (overlap and window size are parameters)
    #handle edge case by just running prediction on rectangular image 
    #for each sliding window image, run inference and get predictions
    #use non max suppresion for overlaps
    
    
    
    # write predictions to format capable of converting to .mat
    # mat_annotations = {"annotations": ["Image_name", "YOLOv8", ["Defect", "Defect", "Defect"], 
    # [[1,0,0],[1,0,0],[1,0,0]], [[12],[15],[17]], [[21,124,12,12],[21,436,4353,23],[234,234,235,64]]]}
    # savemat(filename.mat, mat_annotations)





def main():

    #timing
    start_time = time.perf_counter()
    
    configure()
    model = YOLO(MODEL_PATH)

    #will be .tif file 
    im_path = "/Users/arjunchandra/Desktop/11_X10751_Y19567.(8_112).png"
    im_path = "/Users/arjunchandra/Desktop/reshape.png"
    #im_path = "/Users/arjunchandra/Desktop/none.png"

    get_local_pred(model, im_path)
    get_roboflow_pred(im_path)
    

    finish_time = time.perf_counter()
    print(f"Inference finished in {finish_time-start_time} seconds")


if __name__ == "__main__":
    """Run from Command Line"""
    main()
