import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgHarris = gray_image.copy()

blockSize = 2
aperture_size = 3
k = 0.04

dst = cv2.cornerHarris(gray_image, blockSize, aperture_size, k)

threshold = 0.04; 

B = 0
G = 0
R = 0
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

