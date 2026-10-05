import cv2 
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("8bit-image.png", cv2.IMREAD_GRAYSCALE)

counts, _ = np.histogram(image.ravel(), bins=256, range=(0,256))
pdf = counts / (image.shape[0] * image.shape[1])

cdf = np.cumsum(pdf)

lut = np.round(cdf * 255).astype(np.uint8)

equalized_image = lut[image]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
axes[0, 0].set_title("Original image")
axes[0, 0].axis("off")

axes[0, 1].imshow(equalized_image, cmap="gray", vmin=0, vmax=255)
axes[0, 1].set_title("Equalized image")
axes[0, 1].axis("off")

axes[1, 0].hist(image.ravel(), bins=256, range=(0, 256))
axes[1, 0].set_title("Original histogram")

axes[1, 1].hist(equalized_image.ravel(), bins=256, range=(0, 256))
axes[1, 1].set_title("Equalized histogram")

for ax in axes[1]:
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Number of pixels")
    ax.set_xlim(0, 255)

plt.tight_layout()
plt.show()

print(pdf.sum())