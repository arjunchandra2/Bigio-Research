"""
Running model inference through Roboflow API
"""

from roboflow import Roboflow
from dotenv import load_dotenv
import os


def configure():
    load_dotenv()

def main():
    configure()

    rf = Roboflow(api_key=os.getenv('api_key'))

    #print(rf.workspace().projects())
    project = rf.workspace().project("defect-training-5-3")

    model = project.version(4).model

    print(model)

    # infer on a local image
    print(model.predict("/Users/arjunchandra/Desktop/School/Junior/Bigio Research/Dataset/results/train/images/11_X10751_Y19567.(9_0).png", confidence=40, overlap=30).json())

    # visualize your prediction
    # model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())



main()