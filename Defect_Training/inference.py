"""
Running model inference through Roboflow API
"""

from roboflow import Roboflow
from dotenv import load_dotenv
import os


def configure():
    """
    Wrapper for loading environment 
    """
    load_dotenv()



def main():
    configure()

    rf = Roboflow(api_key=os.getenv('api_key'))

    #print(rf.workspace().projects())
    project = rf.workspace().project("defect-training-5-3")

    model = project.version(4).model

    print(model)

    im_path = "/Users/arjunchandra/Desktop/11_X10751_Y19567.(8_112).png"

    # infer on a local image
    print(model.predict(im_path, confidence=40, overlap=30).json())

    # visualize your prediction
    model.predict(im_path, confidence=40, overlap=30).save("/Users/arjunchandra/Desktop/prediction_test.jpg")

    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())



if __name__ == "__main__":
    """Run from Command Line"""
    main()
