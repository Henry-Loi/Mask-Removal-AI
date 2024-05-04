import cv2
import numpy as np
import pandas as pd


def add_face_mask(original_image_path, face_mask_path, bounding_boxes):
    # Load the original image and the face mask image
    original_image = cv2.imread(original_image_path)
    face_mask = cv2.imread(face_mask_path, cv2.IMREAD_UNCHANGED)

    # Iterate through each bounding box
    for x, y, w, h in bounding_boxes:
        # Resize the face mask to fit the bounding box
        resized_mask = cv2.resize(face_mask, (w, h//2))
        resized_mask[:, :, 3] = resized_mask[:, :, 3] / 255.0
        original_image[y+h-h//2:y+h, x:x+w, 0:3] = resized_mask[:, :,0:3]*resized_mask[:, :,3:] + original_image[y+h-h//2:y+h, x:x+w,0:3]*(1.0 - resized_mask[:, :,3:])

    # Save the modified image
    cv2.imwrite(original_image_path, original_image)

# Read the CSV file
df = pd.read_csv('faceDataset/faces.csv')


for index, row in df.iterrows():
    image_name = row['image_name']
    original_image = cv2.imread('faceDataset/images/'+image_name)
    cv2.imwrite('faceDataset/images/'+image_name[:-4]+"_modified.jpg", original_image)

yolo_data = {}
for index, row in df.iterrows():
    image_name = row['image_name']
    width = row['width']
    height = row['height']
    x0 = row['x0']
    y0 = row['y0']
    x1 = row['x1']
    y1 = row['y1']

    # print(image_name, width, height, x0, y0, x1, y1)
    bounding_boxes = [(x0,y0,x1-x0,y1-y0)]  # Example bounding boxes
    add_face_mask('faceDataset/images/'+image_name[:-4]+"_modified.jpg", "face_mask.png", bounding_boxes)

    # Calculate the center coordinates and dimensions of the bounding box
    x_center = (x0 + x1) / (2 * width)
    y_center = (y0 + y1) / (2 * height)
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height

    with open('faceDataset/labels/'+image_name[:-4]+'_modified.txt', 'a+') as f:
        f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")

# 0: masked, 1:not masked

