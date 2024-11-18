import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./ATU_GRAY.jpg', gray_image)

rows = 2
cols = 2
plt.subplot(rows, rows,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title("Gray Scale"), plt.xticks([]), plt.yticks([])
plt.show()