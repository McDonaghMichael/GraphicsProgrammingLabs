import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./ATU_GRAY.jpg', gray_image)


rows = 2
cols = 2
plt.subplot(rows, rows,1),plt.imshow(cv2.cvtColor(img, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title("Gray Scale"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,3),plt.imshow(cv2.GaussianBlur(gray_image,(3, 3),3), cmap = 'gray')
plt.title("3x3"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,4),plt.imshow(cv2.GaussianBlur(gray_image,(13, 13),3), cmap = 'gray')
plt.title("13x13"), plt.xticks([]), plt.yticks([])

sobelHorizontal = cv2.Sobel(imgIn,cv2.CV_64F,1,0,ksize=5) # x dir
sobelVertical = cv2.Sobel(imgIn,cv2.CV_64F,0,1,ksize=5) # y dir


plt.show()