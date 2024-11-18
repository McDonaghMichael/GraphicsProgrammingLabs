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

plt.subplot(cols, cols,3),plt.imshow(cv2.GaussianBlur(img,(3, 3),3), cmap = 'gray')
plt.title("Blur Scale"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,4),plt.imshow(cv2.GaussianBlur(img,(13, 13),3), cmap = 'gray')
plt.title("Blur Scale"), plt.xticks([]), plt.yticks([])


plt.show()