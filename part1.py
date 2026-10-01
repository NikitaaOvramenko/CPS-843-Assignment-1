import cv2 
import numpy as np

image = cv2.imread("8bit-image.png")

float_image = image.astype(np.float32) / 255.0

noise = np.random.normal(loc=0.0,scale=0.01,size=float_image.shape)

cv2.imshow("Image Display",image)
cv2.imshow("Image Display 2",float_image)

result = float_image + noise

cv2.imshow("Image Display 3",result)

print("new feature")
cv2.waitKey(0)
cv2.destroyAllWindows()