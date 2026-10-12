import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('images/img1.png')

#cv.imshow('img1', img)
#img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# cas = cv.Canny(img,125,175)
# cv.imshow('cas', cas)
# cv.waitKey(0)

# plt.figure(figsize=(10, 8))a
# plt.imshow(img_rgb)
# plt.axis("off")


# 1. Crop at the smaller green square

# 2. Identify countours
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
con = cv.Canny(img,100,200)
cv.imshow('con', con)

contours, hierarchies = cv.findContours(con,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
#print('f'len(contours)) #number of contours found


cv.waitKey(0)