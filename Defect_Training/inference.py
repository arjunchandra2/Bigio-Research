"""
Running model inference through Roboflow API/local inference
and stitching predictions for .tif image into .mat format 
"""

from roboflow import Roboflow
from ultralytics import YOLO
from skimage import io
from PIL import Image
from dotenv import load_dotenv
from scipy.io import savemat
from annotation import Annotation
import numpy as np
import time
import os


#path to YOLO model 
MODEL_PATH = '/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Bigio-Research/Defect_Training/best.pt'
#confidence threshold for detections (0 to 1)
CONFIDENCE = 0.2
#window size
WINDOW_SIZE = 300
#window overlap for sliding window(0 to 1)
WINDOW_OVERLAP = 0.3

#class encodings 
CLASS_ENCODING = {0: 'Defect', 1: 'Swelling', 2: 'Vesicle'}
#color encodings
COLOR_ENCODING = {'Defect': [1,0,0], 'Swelling': [0,1,0], 'Vesicle': [0,0,1]}


def configure():
    """
    - Wrapper for loading environment 
    - Also loads and returns local YOLO model
    """
    load_dotenv()

    return YOLO(MODEL_PATH)

    
def process_image(image_path):
    """
    - Returns list of Pillow images containing z-planes
    """
    im = io.MultiImage(image_path)
    im_frames = im[0]
    pil_frames = []
    #skimage -> PIL for easier cropping and displaying 
    for im in im_frames:
        image = Image.fromarray(im, mode="RGB")
        pil_frames.append(image)
        
    return pil_frames


def get_mat(im_path):
    """
    Creates well-formatted dictionary for .mat annotations
    """
    #ex:
    # mat_annotations = {"annotations": ["Image_name", "YOLOv8", ["Defect", "Defect", "Defect"], 
    #[[1,0,0],[1,0,0],[1,0,0]], [[12],[15],[17]], [[21,124,12,12],[21,436,4353,23],[234,234,235,64]]]}
    # savemat(filename.mat, mat_annotations)
    
    mat_annotations = {"annotations": [im_path, "YOLOv8", [], [], [], []]}

    for annotation in Annotation.annotations:
        for i in range(len(annotation.bboxes)):
            #format xywh so xy is in reference to entire image instead of subimage
            annotation.bboxes[i][0] += annotation.window_left
            annotation.bboxes[i][1] += annotation.window_upper

            mat_annotations['annotations'][2].append(annotation.cls_names[i])
            mat_annotations['annotations'][3].append(list(map(float, COLOR_ENCODING[annotation.cls_names[i]])))
            mat_annotations['annotations'][4].append([annotation.z_plane])
            mat_annotations['annotations'][5].append(annotation.bboxes[i])

    return mat_annotations


#Decorator for model inference functions get_local_pred and get_roboflow_pred
#Decorated function should return well-formatted predictions
def inference(get_pred):

    def inference_wrapper(*args, **kwargs):
        """
        - Runs model inference on .tif image and returns formatted .mat file
        for viewing model annotations in Matlab software
        - args[0] should be image_path
        - args[1] is optionally the model to use
        """
        image_path = args[0]
        z_stack = process_image(image_path)

        #this loop can be changed to skip inference on blurry frames 
        for z in range(len(z_stack)):
            #PIL image for current plane in z_stack
            z_plane = z_stack[z]
            width, height = z_plane.size

            #initalize sliding window
            window_left = 0
            window_right = WINDOW_SIZE
            window_upper = 0 
            window_lower = WINDOW_SIZE

            while window_upper < height:

                window_left = 0
                window_right = WINDOW_SIZE

                while window_left < width:

                    #crop image at window location
                    window = z_plane.crop((window_left, window_upper, window_right, window_lower))
           
                    #get predictions, get_pred return values should be ordered numpy arrays 
                    if len(args) == 2:
                        bboxes, cls_names, confs = get_pred(window, args[1])
                        #create annotation if any predictions are made
                        if len(bboxes) > 0:
                            #NMS here - to be implemented
                            annotation = Annotation(bboxes, cls_names, confs, z+1, window_left, window_upper)  
                                   
                    else:
                        raise NotImplementedError

                    #slide window, ensure right side does not exceed image width
                    #with overlap on, window behavior at edges of image might not be desirable - could add a check
                    #to break loop here if window_right == width
                    if window_right == width:
                        break
                    window_left = window_left + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP
                    window_right = min(width, window_right + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP)

                #same note about overlap 
                if window_lower == height:
                    break
                window_upper = window_upper + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP
                window_lower = min(height, window_lower + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP)
        
        #Create dictionary from Annotations  
        mat_dict = get_mat(image_path)
     
        #Create .mat file and save
        savemat(image_path + '.mat', mat_dict)
        
        #Clear annotations for image
        Annotation.annotations = []

    return inference_wrapper


@inference
def get_local_pred(image, model):
    """
    - Image inference from local Yolov8 model 
    - Image can be any of YOLO accepted formats: 
    https://docs.ultralytics.com/modes/predict/#inference-sources
    - Bottleneck for time efficiency: ~25s for 100 subimages w/o batch processing
    - Can speed up with batch processing (not supported yet) and/or changing device to gpu
    """ 
    #imgsz = (width, height), recommended to resize to (640,640) -> seems to work fine even for rectangular images
    #resizing maintains aspect ratio using rescale and pad and maintains multiple of 32 (network stride)
    results = model.predict(source=image, conf=CONFIDENCE, imgsz=640, iou=0.7, device='cpu')

    bboxes = []
    cls_names = []
    confs = []

    #only a single result if no batch inference
    for result in results:
        preds = result.boxes.numpy()
        bboxes = preds.xyxy.astype('int32')
        cls_names = preds.cls
        confs = preds.conf
    
    #cls_id -> cls_name
    cls_names = [CLASS_ENCODING[cls_id] for cls_id in cls_names]
    #xyxy -> xywh
    bboxes = [[bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] for bbox in bboxes]

    return bboxes, cls_names, confs 



#Not yet supported for full inference in case of rate limits
def get_roboflow_pred(im_path):
    """
    - Get model predictions via Roboflow API
    - Using YOLO-NAS model: 0.425 mAP
    - Image format must be path (does not support PIL etc.)
    """
    rf = Roboflow(api_key=os.getenv('api_key'))
    #print(rf.workspace().projects())
    project = rf.workspace().project("defect-training-5-3")
    model = project.version(4).model
    print(model)

    # infer on a local image - overlap set to 70%
    # predictions are (center_x, center_y, width, height)
    print(model.predict(im_path, confidence=CONFIDENCE, overlap=70).json())

    # visualize your prediction
    #model.predict(im_path, confidence=40, overlap=30).save("/Users/arjunchandra/Desktop/prediction_test.jpg")
    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
    



def main():

    #timing
    start_time = time.perf_counter()
    model = configure()

    #path to .tif image (.mat file will be saved in same directory)
    im_path = "/Users/arjunchandra/Desktop/11_X10821_Y18288.tif"
    
    get_local_pred(im_path, model)
    #get_roboflow_pred(im_path)

    finish_time = time.perf_counter()
    print(f"\nInference finished in ~{(finish_time-start_time)//60} minutes")



if __name__ == "__main__":
    """Run from Command Line"""
    main()
