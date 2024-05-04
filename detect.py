from ultralytics import YOLO

import cv2

class Prediction:
    def __init__(self):
        self. model = YOLO("./runs/detect/train10/weights/best.pt") 

    def predict(self, image):
        return self.model.predict(image,verbose=False)


prediction = Prediction()

# frame = cv2.imread("faceDataset/images/00003121.jpg")
# frame = cv2.imread("faceDataset/images/00003517.jpg")
# frame = cv2.imread("faceDataset/images/00000003.jpg")
frame = cv2.imread("datasets/valid/images/1152x768_246964803156_jpg.rf.2f7be23abc3c85e25ee3c3308988a6fb.jpg")
results = prediction.predict(frame)

for r in results:
    im_array = r.plot()
    cv2.imshow("frame", im_array)


cv2.waitKey(0)
cv2.destroyAllWindows()
