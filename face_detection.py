from ultralytics import YOLO

import pandas as pd
import cv2
import numpy as np
import os

# dataset: https://universe.roboflow.com/jonathan-chandra/face-mask-object-detection

# Load a model
# model = YOLO("yolov8n.pt")  # load a pretrained model 
model = YOLO("./runs/detect/train10/weights/best.pt") 
# Train the model
results = model.train(data="./datasets/data.yaml", batch=8, epochs=1, plots=True, workers = 4)
