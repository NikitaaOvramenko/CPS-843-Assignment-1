import cv2 
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("8bit-image.png", cv2.IMREAD_GRAYSCALE)

float_image = image.astype(np.float32) / 255.0

k = 100
variance = 0.01
sigma = np.sqrt(variance)


results = []

for i in range(k):

    noise = np.random.normal(loc=0.0,scale=sigma,size=float_image.shape)
    result = float_image + noise
    results.append(result)



meanImage = sum(results)/k
print(meanImage)


MSE = np.mean(np.square(float_image - meanImage))

print("MSE:",MSE)


cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Original image", 800, 600)
cv2.imshow("Original image", image)

cv2.namedWindow("One noisy image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("One noisy image", 800, 600)
cv2.imshow("One noisy image", np.clip(results[0], 0, 1))

label = f"Averaged image | K={k} | MSE={MSE:.6f}"

cv2.namedWindow(label, cv2.WINDOW_NORMAL)
cv2.resizeWindow(label, 800, 600)
cv2.imshow(label, np.clip(meanImage, 0, 1))


cv2.waitKey(0)
cv2.destroyAllWindows()

x = [5,10,50,100]
y = [0.001987,0.001000,0.000202,MSE]

plt.plot(x,y)

plt.title("Calculated MSE as a function of K")
plt.xlabel("K")
plt.ylabel("MSE")
plt.show()