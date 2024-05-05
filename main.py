from ultralytics import YOLO

import cv2

class Prediction:
    def __init__(self):
        self. model = YOLO("./runs/detect/train12/weights/best.pt") 

    def predict(self, image):
        return self.model.predict(image,verbose=False)


prediction = Prediction()

#change family photo here to change the input
# family_photo = cv2.imread("faceDataset/images/00003121_modified.jpg")
# family_photo = cv2.imread("faceDataset/images/00003517_modified.jpg")
# family_photo = cv2.imread("datasets/valid/images/1152x768_246964803156_jpg.rf.2f7be23abc3c85e25ee3c3308988a6fb.jpg")
family_photo = cv2.imread("test/masked_girls.jpg")
results = prediction.predict(family_photo)

for r in results:
    for box in r.boxes.data:# for each bounding box of face
        if box[5] == 0: #if masked
            x1, y1, x2, y2 = box[:4]
            x1=int(x1.item())
            y1=int(y1.item())
            x2=int(x2.item())
            y2=int(y2.item())

            # show that face
            cv2.imshow("frame", family_photo[y1:y2, x1:x2])
            # @Henry: you can do image registration here

            cv2.waitKey(0)


# show the family photo with bounding box of masked faces only
for r in results:
    for box in r.boxes.data:
        if box[5] == 0:
            x1, y1, x2, y2 = box[:4]
            x1=int(x1.item())
            y1=int(y1.item())
            x2=int(x2.item())
            y2=int(y2.item())
            cv2.rectangle(family_photo, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("Family Photo", family_photo)
cv2.waitKey(0)

# show the family photo with all bounding boxes
for r in results:
    for box in r.boxes.data:
        x1, y1, x2, y2 = box[:4]
        x1=int(x1.item())
        y1=int(y1.item())
        x2=int(x2.item())
        y2=int(y2.item())
        if box[5] == 0:
            cv2.rectangle(family_photo, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            cv2.rectangle(family_photo, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Family Photo", family_photo)
cv2.waitKey(0)

cv2.destroyAllWindows()
