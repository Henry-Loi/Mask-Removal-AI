import pandas as pd

# Read the CSV file
df = pd.read_csv('faceDataset/faces.csv')
print()

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

    # Calculate the center coordinates and dimensions of the bounding box
    x_center = (x0 + x1) / (2 * width)
    y_center = (y0 + y1) / (2 * height)
    box_width = (x1 - x0) / width
    box_height = (y1 - y0) / height

    with open('face_yolo_dataset/'+image_name[:-4]+'.txt', 'a+') as f:
        f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
