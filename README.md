# Mask-Removal-AI

### dataset not pushed
#### The original kaggle csv dataset
i put https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection/data in a folder named `faceDataset`

it consists of a `faceDataset/faces.csv` and a `images` directory with 3000+ images

#### Face mask yolo dataset
i downloaded another dataset https://universe.roboflow.com/jonathan-chandra/face-mask-object-detection from detecting face mask. it has a yolo format so i need to call its directory `datasets`. i want to steal a pre-trained yolo model but i only got a dataset. 

#### The kaggle yolo dataset
##### csv_to_yolo.py
i use `csv_to_yolo.py` to convert `faceDataset/faces.csv` into `labels` of yolo format so we can use those later?
The labels are stored in `faceDataset/labels`

##### add_face_mask.py
i use `add_face_mask.py` to add face masks (`face_mask.png`) into the images of `faceDataset` their file names ends with `_modified`

### yolo model
#### face_detection.py
i trained the yolo model with `face_detection.py` for 8 epochs (i found that i forgot to load the previous trained model for the first few models :P) and it automatically stored the resulting model in `runs/detect/train10/weights/best.pt` 
>need to change the model inside the code

#### detect.py
to see the result, run `detect.py` 
>need to change the model and image inside