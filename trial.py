
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load images and masks
img1 = cv.imread('images/img1.png')
img2 = cv.imread('images/img2.png')
mask1 = cv.imread('images/mask1.png')
mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY) #convert to grayscale to make it easier to compare later
mask2 = cv.imread('images/mask2.png')
mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)
newImage1 = cv.imread('images/unseen_img1.png')
newImage2 = cv.imread('images/unseen_img2.png')


if any(x is None for x in [img1, img2, mask1, mask2, newImage1, newImage2]): #if not found
    raise cv.error("Files not found!")

# 2. Select the inner green square in each image to look at the region of interest
def crop(img, name):
    x, y, w, h = cv.selectROI(
        name, img, fromCenter=False 
    )
    cv.destroyWindow(name)

    if w == 0 or h == 0: #the rectagle has no width or height
        raise cv.error("Select the region to look at")

    return img[y:y+h, x:x+w], (x, y, w, h)

#crop the two iamges given and the unseen all here to give results later
crop1, region1 = crop(img1, "Select region in image 1")
crop2, region2 = crop(img2, "Select region in imgage 2")

## The masks are already cropped but may not align with user cropping so make sure it matches
def cropMask(mask, roi):
    x, y, w, h = roi
    return mask[y:y+h, x:x+w]

maskCrop1 = cropMask(mask1, region1)
maskCrop2 = cropMask(mask2, region2)


# 3. Generate contours of the images and extract features
def getShapes(img, mask=None): #the two images have the mask to detect barnacles but the new images don't
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200) #get the edges using Canny

    #trace boundaries using the edges to find shapes
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # For trianing data: the mask is known for img1 and img2
    # assume darker pixels are the barnacle shape outlines
    if mask is not None:
        outline = (mask < 200).astype(np.uint8) #determine if there is a barnacle based on pixel brightness since outline is darker
        outline = cv.dilate(outline, np.ones((7, 7), np.uint8)) #make the outline on masks thicker so its easier to overlap and compare with later
    #Extract features for each contour found: area, circularity, mean color, mean gray brightness, and the number of edges detected
    data = []
    features = []
    label = []

    for contour in contours:
        area = cv.contourArea(contour)

        # Ignore small contours and very large regions so doesnt double count
        if area < 30 or area > img.shape[0] * img.shape[1] * 0.1:
            continue

        x, y, w, h = cv.boundingRect(contour) #make rectangle so easier to capture

        # Features
        oneShape = img[y:y+h, x:x+w] #color
        grayPart = gray[y:y+h, x:x+w] #gray
        edgePart = edges[y:y+h, x:x+w] #edge or not

        perimeter = cv.arcLength(contour, True) #to make sure denom is not 0

        circularity = 0
        if perimeter > 0:
            circularity = 4*np.pi*area/(perimeter**2) #look at how close to a circle

        features.append([
            area,
            circularity,
            *np.mean(oneShape, axis=(0, 1)),
            np.mean(grayPart),
            np.mean(edgePart > 0)
        ])

        data.append(contour)

        # Determine the y labael based on overlap with annotated outlines
        if mask is not None:
            outlineRegion = outline[y:y+h, x:x+w] #get the appropridate region of mask for each contour
            overlap = np.mean(outlineRegion > 0) #see if there is a outline line there
            label.append(1 if overlap > 0.05 else 0)

    return data, np.array(features), (
        np.array(label) if mask is not None else None
    )

contours1, X1, y1 = getShapes(crop1, maskCrop1)
contours2, X2, y2 = getShapes(crop2, maskCrop2)

# 4. Train on image 1 and test on image 2 to evaluate performance

#Implement random forest classifier
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=1)

model.fit(X1, y1) #train
pred2 = model.predict(X2) #test

print("\nEvaluation on image 2:")
print(classification_report(
    y2, pred2,
    labels=[0, 1],
    target_names=['Background', 'Barnacle'],
    zero_division=0
))
#show visualization of image 2
showLabels = crop2.copy()

labeledBarnacles = [
    contour for contour, label in zip(contours2, pred2)
    if label == 1
]

cv.drawContours(showLabels, labeledBarnacles, -1, (255, 0, 0), 2)

cv.imshow(f"Predicted barnacles: {len(labeledBarnacles)}",showLabels)
print(f"Predicted barnacles in image 2: {len(labeledBarnacles)}")
cv.waitKey(0)

# 5. Train on both labeled images (1 and 2) for the unseen image detection
xTrain = np.vstack([X1, X2])
yTrain = np.concatenate([y1, y2])

#look at mean characteristic values
model.fit(xTrain, yTrain)
featureNames = [
    "Area",
    "Circularity",
    "Mean Blue",
    "Mean Green",
    "Mean Red",
    "Mean Grayscale (Brightness)",
    "Edge Density (Percent edges present)"
]
for name, meanValue in zip(featureNames, np.mean(X1, axis=0)):
    print(f"{name}: {meanValue:.2f}")
print("\n")
for name, meanValue in zip(featureNames, np.mean(X2, axis=0)):
    print(f"{name}: {meanValue:.2f}")

# Prepare the candidates for the unseen images
cropNew1, regionNew1 = crop(newImage1, "Select square in this new image 1")
cropNew2, regionNew2 = crop(newImage2, "Select square in this new image 2")

 
contoursNew1, featuresNew1, yLabel1 = getShapes(cropNew1)
predNew1 = model.predict(featuresNew1)

contoursNew2, featuresNew2, yLabel2 = getShapes(cropNew2)
predNew2 = model.predict(featuresNew2)

# 6. Count predicted barnacle candidates
barnacleContours1 = [
    contour for contour, label in zip(contoursNew1, predNew1)
    if label == 1
]
barnacleContours2 = [
    contour for contour, label in zip(contoursNew2, predNew2)
    if label == 1
]
print("\nPredicted barnacles in unseen img1:", len(barnacleContours1))
print("\nPredicted barnacles in unseen img2:", len(barnacleContours2))

# 7. Visualize detections
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
result1 = cropNew1.copy()
result2 = cropNew2.copy()
cv.drawContours(result1, barnacleContours1, -1, (255, 0, 0), 2) 
cv.drawContours(result2, barnacleContours2, -1, (255, 0, 0), 2) 

axes[0].imshow(cv.cvtColor(result1, cv.COLOR_BGR2RGB))
axes[0].set_title(f"Predicted barnacles: {len(barnacleContours1)}")

axes[1].imshow(cv.cvtColor(result2, cv.COLOR_BGR2RGB))
axes[1].set_title(f"Predicted barnacles: {len(barnacleContours2)}")

for ax in axes:
    ax.axis("off")

plt.suptitle("Unseen image predictions")
plt.tight_layout()
plt.show()