import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU1.jpg")


rows = 3 # How much rows in our popup
cols = 3 # How much columns in our popup

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgHarris = img.copy()
blockSize = 2
aperture_size = 3
k = 0.04
dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)
threshold = 0.04; 
B = 0
G = 0
R = 255

for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

imgShiTomasi = img.copy()

maxCorners = 100
qualityLevel = 0.01
minDistance = 10
corners = cv2.goodFeaturesToTrack(gray_image,maxCorners,qualityLevel,minDistance)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(B, G, R),-1)

# Original Image
plt.subplot(cols, cols,1)
plt.imshow(img, cmap = 'gray')
plt.title("Original")
plt.xticks([])
plt.yticks([])

# Gray Scale Image
plt.subplot(cols, cols,2)
plt.imshow(gray_image, cmap = 'gray')
plt.title("Gray Scale")
plt.xticks([])
plt.yticks([])

# Harris Corners
plt.subplot(cols, cols,3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.xticks([])
plt.yticks([])

plt.subplot(cols, cols,4)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title("Shi Tomasi")
plt.xticks([])
plt.yticks([])

plt.show()