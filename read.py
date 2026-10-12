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
#convert to hsv to take into account all shades of green without 
#disturbing the others in RGB - multiple combination with green overlap
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#has the hue (color), saturation (how vivid), and value (brightness)

#find green color ranges:
lower = np.array([30,40,40])
higher = np.array([100, 255, 255])

#now only look at the zone inside the green (crop)
crop = cv.inRange(hsv, lower, higher)
#cv.imshow("cropped", crop) #just the green parts
#cv.waitKey(0)

# Get coordinates of green pixels
# Select the inner square with your mouse
x, y, w, h = cv.selectROI(
    "Select inner square",
    img,
    showCrosshair=True,
    fromCenter=False
)

# Crop the original image
if w > 0 and h > 0:
    cropped = img[y:y+h, x:x+w]

    cv.imshow("Barnacles inside square", cropped)
    cv.waitKey(0)

# box = cv.findNonZero(crop) #find the box
# if(box is not None):
#     x,y,w,h = cv.boundingRect(box)
#     #print("Bounding box:", x, y, w, h)

#         # Display the detected outer boundary
#     debug = img.copy()
#     cv.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv.imshow("Detected outer boundary", debug)
#     cv.waitKey(0)

#     #now crop
#     margin = 5
#     cropped = img[y+margin: y+h-margin, x+margin:x+w-margin]

#     cv.imshow("cropped image", cropped)
#     cv.waitKey(0)
# else:
#     print("Issue: no green border found!")

#2. Identify countours
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
con = cv.Canny(cropped,125,175)
cv.imshow('con', con)
cv.waitKey(0)

contours, hierarchies = cv.findContours(con,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours found') #number of contours found

# Draw detected contours on the original crop
result = cropped.copy()
cv.drawContours(result, contours, -1, (0, 0, 255), 2)
cv.imshow("contours", result)
cv.waitKey(0)


# #3 Use the brightness and lightness in the canny iamge for detection
# gray = cv.cvtColor(cropped, cv.COLOR_BGR2GRAY)

# # Identify locally brighter or darker regions because it varies
# binary = cv.adaptiveThreshold(gray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 3)  
# cv.imshow("binary", binary)
# cv.waitKey(0)


#Now count:

count = 0

for contour in contours:
    area = cv.contourArea(contour)
        # Ignore tiny contours
    if area < 30:
        continue
    cv.drawContours(result, [contour], -1, (0, 255, 0), 1)
    count += 1


print("Number of barnacles:", count)

#check:


# cv.drawContours(result, count,-1,(0, 255, 0),2) 
# cv.imshow("contours", result)
# cv.waitKey(0)