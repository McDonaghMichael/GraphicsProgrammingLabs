import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("ATU.jpg")

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

harris_image = gray_image.copy()

blockSize = 2
aperture_size = 3
k = 0.04

dst = cv2.cornerHarris(harris_image, blockSize, aperture_size, k)