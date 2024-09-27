"""
Running model inference through Roboflow API/local inference and stitching predictions for .tif image into .mat format 
"""

#Roboflow model unused as of 6/9/24
from roboflow import Roboflow
from ultralytics import YOLO
from skimage import io
from PIL import Image
import cv2
from dotenv import load_dotenv
from scipy.io import savemat
from annotation import Annotation
import numpy as np
import time
import os
import sys
sys.path.insert(0, "/Users/arjunchandra/Desktop/School/Research/Bigio Research/Bigio-Research")
import utils


#path to YOLO model 
MODEL_PATH = '/Users/arjunchandra/Desktop/School/Research/Bigio Research/Bigio-Research/Defect_Training/Models/best_2.pt'
#confidence threshold for detections (0-1)
CONFIDENCE_THRESHOLD = 0.255
#non max supression threshold (0-1)
NMS_THRESHOLD = 0.2
#window size: for @inference this is sliding window size, for @inference_grayscale this is subimage size 
WINDOW_SIZE = 600
#subwindow size for @inference_grayscale 
SUB_WINDOW_SIZE = 100
#window overlap for sliding window (@inference) or sliding sub_window (@inference_grayscale). Should be (0-1)
WINDOW_OVERLAP = 0.5

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


def get_mat(im_path):
    """
    - Creates well-formatted dictionary for .mat annotations
    - Performs plane by plane NMS: https://gist.github.com/leandrobmarinho/26bd5eb9267654dbb9e37f34788486b5
    """
    #ex:
    # mat_annotations = {"annotations": ["Image_name", "YOLOv8", ["Defect", "Defect", "Defect"], 
    #[[1,0,0],[1,0,0],[1,0,0]], [[12],[15],[17]], [[21,124,12,12],[21,436,4353,23],[234,234,235,64]]]}
    # savemat(filename.mat, mat_annotations)
    
    mat_annotations = {"annotations": [im_path, "YOLOv8", [], [], [], []]}

   
    for z_plane in Annotation.annotations:
        #aggregate all annotations in each z_plane for NMS
        bboxes = []
        confs = []
        cls_names = []

        for annotation in Annotation.annotations[z_plane]:
            for i in range(len(annotation.bboxes)):
                #format xywh so xy is in reference to entire image instead of subimage
                annotation.bboxes[i][0] += annotation.window_left
                annotation.bboxes[i][1] += annotation.window_upper

                bboxes.append(annotation.bboxes[i])
                confs.append(annotation.confs[i])
                cls_names.append(annotation.cls_names[i])

        #non max supression step
        indices = cv2.dnn.NMSBoxes(bboxes, confs, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        for index in indices: 
            mat_annotations['annotations'][2].append(cls_names[index])
            mat_annotations['annotations'][3].append(list(map(float, COLOR_ENCODING[cls_names[index]])))
            mat_annotations['annotations'][4].append([z_plane])
            mat_annotations['annotations'][5].append(bboxes[index])
            

    return mat_annotations


def inference_grayscale(get_pred):
    """
    - Decorator for model inference functions get_local_pred and get_roboflow_pred
    - Decorated function should return well-formatted predictions
    - Image is converted to grayscale before inference
    - Tiles the .tif image into subimages and saves a .mat file for each subimage in the save directory along with the subimage. 
    """

    def sub_image_inference(im_path, save_dir, model, image, i, z, window_left, window_upper):
        """
        - Slides sub_window across subimage and populates Annotation class with predictions
        - Only supports YOLO model inference
        """
        #initalize sliding window
        window_left = 0
        window_right = SUB_WINDOW_SIZE
        window_upper = 0 
        window_lower = SUB_WINDOW_SIZE

        width, height = image.size

        while window_upper < height:

                window_left = 0
                window_right = SUB_WINDOW_SIZE
                while window_left < width:
                    #crop image at window location
                    window = image.crop((window_left, window_upper, window_right, window_lower))

                    bboxes, cls_names, confs = get_pred(window, save_dir, model)
                  
                    #create annotation if any predictions are made
                    if len(bboxes) > 0:
                        #save annotation with z_plane = 1 since it is not a z-stack
                        annotation = Annotation(bboxes, cls_names, confs, 1, window_left, window_upper)  
                    
                    #window behavior at edges of image might not be desirable 
                    if window_right == width:
                        break

                    #slide window to the right
                    window_left = window_left + SUB_WINDOW_SIZE - SUB_WINDOW_SIZE*WINDOW_OVERLAP
                    window_right = min(width, window_right + SUB_WINDOW_SIZE - SUB_WINDOW_SIZE*WINDOW_OVERLAP)

                #window behavior at edges of image might not be desirable 
                if window_lower == height:
                    break

                #slide window down
                window_upper = window_upper + SUB_WINDOW_SIZE - SUB_WINDOW_SIZE*WINDOW_OVERLAP
                window_lower = min(height, window_lower + SUB_WINDOW_SIZE - SUB_WINDOW_SIZE*WINDOW_OVERLAP)

        im_path_tail = os.path.split(im_path)[1]
        save_path = save_dir + '/' + im_path_tail[:-4] + f'_({z}_{i}).png'

        #Create dictionary from Annotations  
        mat_dict = get_mat(save_path)
        #Create .mat file and save image 
        image.save(save_path)   
        savemat(save_path + '.mat', mat_dict)

        #Clear annotations for image
        Annotation.annotations = {}

        print(f"Saved image: {save_path}")
                                   

    def inference_grayscale_wrapper(*args, **kwargs):
        """
        - Runs model inference on .tif image and returns formatted .mat file containing annotations
        - args[0] should be image_path
        - args[1] should be save directory
        - args[2] is the model to use
        """
        im_path = args[0]
        save_dir = args[1]
        model = args[2]

        z_stack = utils.process_image(im_path)

        #skip blurry planes, only use planes 8-21 as in training data 
        for z in range(7,len(z_stack)-4):
            #subimage index
            i = 0
            #PIL image for current plane in z_stack
            z_plane = z_stack[z]

            #convert to grayscale
            z_plane = utils.convert_to_grayscale(z_plane)
            width, height = z_plane.size

            #initalize sliding window
            window_left = 0
            window_right = WINDOW_SIZE
            window_upper = 0 
            window_lower = WINDOW_SIZE

            #The logic is slightly different from @inference because we are not trying to make predictions for the entire image. 
            #We are making predictions for subimages of the image and only want the square Window_SIZE x Window_SIZE subimages
            #and there should be no overlap between subimages. 
            while window_lower < height:

                window_left = 0
                window_right = WINDOW_SIZE

                while window_right < width:

                    #crop image at window location
                    window = z_plane.crop((window_left, window_upper, window_right, window_lower))

                    #only support local YOLO model inference  
                    if len(args) == 3:
                        sub_image_inference(im_path, save_dir, model, window, i, z+1, window_left, window_upper)
                        i +=1       
                    else:
                        raise NotImplementedError

                    #slide window to the right
                    window_left = window_left + WINDOW_SIZE 
                    window_right = window_right + WINDOW_SIZE 

                #slide window down
                window_upper = window_upper + WINDOW_SIZE 
                window_lower = window_lower + WINDOW_SIZE 

    return inference_grayscale_wrapper



def inference(get_pred):
    """
    - Decorator for model inference functions get_local_pred and get_roboflow_pred
    - Decorated function should return well-formatted predictions
    - Creates a single .mat file for the entire .tif image and saves it in the same directory as the image 
    """

    def inference_wrapper(*args, **kwargs):
        """
        - Runs model inference on .tif image and returns formatted .mat file containing annotations
        - args[0] should be image_path/image
        - args[1] should be save directory
        - args[2] is optionally the model to use
        """
        im_path = args[0]
        z_stack = utils.process_image(im_path)

        #this loop can be changed to skip inference on blurry planes
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
                    if len(args) == 3:
                        bboxes, cls_names, confs = get_pred(window, args[1], args[2])
                        #create annotation if any predictions are made
                        if len(bboxes) > 0:
                            annotation = Annotation(bboxes, cls_names, confs, z+1, window_left, window_upper)  
                                    
                    else:
                        raise NotImplementedError

                    #window behavior at edges of image might not be desirable 
                    if window_right == width:
                        break

                    #slide window to the right
                    window_left = window_left + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP
                    window_right = min(width, window_right + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP)

                #window behavior at edges of image might not be desirable 
                if window_lower == height:
                    break

                #slide window down
                window_upper = window_upper + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP
                window_lower = min(height, window_lower + WINDOW_SIZE - WINDOW_SIZE*WINDOW_OVERLAP)
        
        #Create dictionary from Annotations  
        mat_dict = get_mat(im_path)
     
        #Create .mat file and save
        savemat(im_path + '.mat', mat_dict)
        
        #Clear annotations for image
        Annotation.annotations = {}

    return inference_wrapper


@inference_grayscale
def get_local_pred(image, save_dir, model):
    """
    - Image inference from local Yolov8 model 
    - YOLO accepted formats: https://docs.ultralytics.com/modes/predict/#inference-sources
    - Bottleneck for time efficiency: ~25s for 100 subimages w/o batch processing
    - Can speed up with batch processing (not supported yet) and/or changing device to gpu
    """ 
    #imgsz = (width, height), recommended to resize to (640,640) -> seems to work fine even for rectangular images
    #resizing maintains aspect ratio using rescale and pad and maintains multiple of 32 (network stride)
    results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD, imgsz=640, iou=NMS_THRESHOLD, device='cpu')

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
def get_roboflow_pred(im_path, save_dir):
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
    print(model.predict(im_path, confidence=CONFIDENCE_THRESHOLD, overlap=100*NMS_THRESHOLD).json())

    # visualize your prediction
    #model.predict(im_path, CONFIDENCE_THRESHOLD=40, overlap=30).save("/Users/arjunchandra/Desktop/prediction_test.jpg")
    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, CONFIDENCE_THRESHOLD=40, overlap=30).json())
    

def main():

    #timing
    start_time = time.perf_counter()
    model = configure()

    #full path to .tif image 
    im_path = "/Users/arjunchandra/Desktop/School/Research/Bigio Research/Annotation Demo/11_X10821_Y18288.tif"
    save_dir = "/Users/arjunchandra/Desktop/School/Research/Bigio Research/grayscale_annotations"

    if save_dir:
        if os.path.exists(save_dir):
            os.system('rm -fr "%s"' % save_dir)
        
        os.mkdir(save_dir)
    
    get_local_pred(im_path, save_dir, model)
    #get_roboflow_pred(im_path, save_dir)

    finish_time = time.perf_counter()
    print(f"\nInference finished in ~{(finish_time-start_time)//60} minutes")



if __name__ == "__main__":
    """Run from Command Line"""
    main()
