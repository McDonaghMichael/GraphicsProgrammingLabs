import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./ATU_GRAY.jpg', gray_image)


blurred_gray = cv2.GaussianBlur(gray_image,(3, 3),3)
blurred_image_13 = cv2.GaussianBlur(gray_image,(13, 13),3)

sobel_x = cv2.Sobel(blurred_image_13,cv2.CV_64F,1,0,ksize=5)
sobel_y = cv2.Sobel(blurred_image_13,cv2.CV_64F,0,1,ksize=5)
canny = cv2.Canny(blurred_gray,100,2)

rows = 3
cols = 3

plt.subplot(rows, rows,1),plt.imshow(cv2.cvtColor(img, 
cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title("Gray Scale"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,3),plt.imshow(blurred_gray, cmap = 'gray')
plt.title("3x3"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,4),plt.imshow(blurred_image_13, cmap = 'gray')
plt.title("13x13"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,5),plt.imshow(sobel_x, cmap = 'gray')
plt.title("Sobel X"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,6),plt.imshow(sobel_y, cmap = 'gray')
plt.title("Sobel Y"), plt.xticks([]), plt.yticks([])

plt.subplot(cols, cols,7),plt.imshow(canny, cmap = 'gray')
plt.title("Canny"), plt.xticks([]), plt.yticks([])


plt.show()