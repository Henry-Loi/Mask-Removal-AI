from deepface import DeepFace

result = DeepFace.verify(img1_path = "img/mymask_with_dica.jpg", img2_path = "img/mymask_with_dica.jpg")

print(result)

# print the image with face detection result with openCv

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img/mymask_with_dica.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
