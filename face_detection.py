from ultralytics import YOLO

import pandas as pd
import cv2
import numpy as np
import os

# dataset: https://universe.roboflow.com/jonathan-chandra/face-mask-object-detection

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model 

# Train the model
results = model.train(data="./datasets/data.yaml", batch=8, epochs=1, plots=True)