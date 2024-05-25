"""
Running model inference through Roboflow API/local inference
and stitching predictions for .tif image into .mat format 
"""

from roboflow import Roboflow
from dotenv import load_dotenv
import os
from scipy.io import savemat


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

    # infer on a local image
    print(model.predict(im_path, confidence=40, overlap=30).json())

    # visualize your prediction
    model.predict(im_path, confidence=40, overlap=30).save("/Users/arjunchandra/Desktop/prediction_test.jpg")

    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
    

def get_local_pred(image):
    """
    - Image inference from local Yolov8 model 
    """
    raise NotImplementedError


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
    
    configure()

    #will be .tif file 
    im_path = "/Users/arjunchandra/Desktop/11_X10751_Y19567.(8_112).png"
    im_path = "/Users/arjunchandra/Desktop/reshape.png"
    
    get_roboflow_pred(im_path)




if __name__ == "__main__":
    """Run from Command Line"""
    main()
