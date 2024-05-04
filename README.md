# Mask-Removal-AI

### dataset not pushed
#### The original kaggle csv dataset
i put https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection/data in a folder named `faceDataset`

it consists of a `faceDataset/faces.csv` and a `images` directory with 3000+ images

#### Face mask yolo dataset
i downloaded another dataset https://universe.roboflow.com/jonathan-chandra/face-mask-object-detection from detecting face mask. it has a yolo format so i need to call its directory `datasets`. i want to steal a pre-trained yolo model but i only got a dataset. 

#### The kaggle yolo dataset
i use `csv_to_yolo.py` to convert `faceDataset/faces.csv` into `labels` of yolo format so we can use those later?
The labels are stored in `face_yolo_dataset`

### yolo model
i trained the yolo model with `face_detection.py` for 3 epoch :P and it automatically stored the resulting model in `runs/detect/train3/weights/best.pt` 

to see the result, run `detect.py` 