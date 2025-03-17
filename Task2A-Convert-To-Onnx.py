#task 2 - Convert to ONNX

#importing packages
from ultralytics import YOLO




# Load the YOLO11 model PT
providedModel = YOLO("Models/yolo11n.pt")

#message
print('Loaded PT Model Successfully')

# export here as ONNX
providedModel.export(format='onnx');

#message
print('Converted Model to ONNX Successfully')
