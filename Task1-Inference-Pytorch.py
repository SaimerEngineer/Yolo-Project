#Task 1 - Reference Pytorch Model

#importing packages
import torch
from ultralytics import YOLO

# Load the YOLO11 model
providedModel = YOLO("Models/yolo11n.pt")

#message
print('Loaded PT Model Successfully')

# Load the image to test
img = 'Images/image.png'

# Perform inference
results = providedModel(img)

#message
print('Performed Inference Successfully')

# Show results
for coordinates in results:
    coordinates.show()

#ran code successfully
print('Code ran successfully')