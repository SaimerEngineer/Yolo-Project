#task 2 - Inference ONNX file to test

#importing packages
from ultralytics import YOLO

# Load the image to test
img = 'Images/image.png'

#load new the model ONNX
convertedONNXModel = YOLO("Models/yolo11n.onnx")

#message
print('Loaded Model Successfully')

# Perform inference
results = convertedONNXModel(img)

#message
print('Printing Results')

# Show results
for coordinates in results:
    coordinates.show()

#ran code successfully
print('Inference Image Successful with ONNX Model')