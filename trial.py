
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
newImage = cv.imread('images/unseen_img1.png')

if any(x is None for x in [img1, img2, mask1, mask2, newImage]): #if not found
    print("Files not found!")

# 2. Select the inner green square in each image to look at the region of interest
def crop(img, name):
    x, y, w, h = cv.selectROI(
        name, img, fromCenter=False #showCrosshair=True,
    )
    cv.destroyWindow(name)

    if w == 0 or h == 0: #the rectagle has no width or height
        print("Select the region to look at")

    cropped = img[y:y+h, x:x+w], (x, y, w, h)
    return cropped

#crop the two iamges given and the unseen all here to give results later
crop1, region1 = crop(img1, "Select region in image 1")
crop2, region2 = crop(img2, "Select region in imgage 2")
cropNew, regionNew = crop(newImage, "Select square in this new image")

## The masks are already cropped so not need that!

def crop_mask(mask, roi):
    x, y, w, h = roi
    return mask[y:y+h, x:x+w]

mask_crop1 = crop_mask(mask1, region1)
mask_crop2 = crop_mask(mask2, region2)

# 3. Generate contours of the images and extract features
def getShapes(img, mask=None): #the two images have the mask to detect barnacles but the new images don't
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shapes = cv.Canny(gray, 100, 200) #get the edges using Canny

    contours, hierarchy = cv.findContours( #trace boundaries using the edges to find shapes
        shapes, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )

    # For trianing data: the mask is known for img1 and img2
    # assumes darker pixels are the barnacle shape outlines
    if mask is not None:
        outline = (mask < 200).astype(np.uint8) #determine if there is a barnacle based on pixel brightness since outline is darker
        outline = cv.dilate(outline, np.ones((7, 7), np.uint8)) #make the region thicker so its easier to overlap and compare with later

    #Extract features for each contour found: area, circularity, average color, deviation in color,
    # average gray color, deviation in that brightness, and the number of edges detected
    data = []
    features = []
    label = []

    for contour in contours:
        area = cv.contourArea(contour)

        # Ignore tiny contours and very large regions
        if area < 30 or area > img.shape[0] * img.shape[1] * 0.1:
            continue

        x, y, w, h = cv.boundingRect(contour) #make rectangle so easier to capture

        if w < 2 or h < 2:
             continue

        # Features: size, shape, color, brightness, and edge density
        shape = img[y:y+h, x:x+w]
        grayPart = gray[y:y+h, x:x+w]
        edgePart = shapes[y:y+h, x:x+w] #edge or not

        perimeter = cv.arcLength(contour, True)

        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2) #look at how close to a circle

        features.append([
            area,
            w / h,
            circularity,
            *np.mean(shape, axis=(0, 1)),
            *np.std(shape, axis=(0, 1)),
            np.mean(grayPart),
            np.std(grayPart),
            np.mean(edgePart > 0)
        ])

        data.append(contour)

            # Determine the y labael based on overlap with annotated outlines
        if mask is not None:
            outlineRegion = outline[y:y+h, x:x+w]
            overlap = np.mean(outlineRegion > 0)
            label.append(1 if overlap > 0.02 else 0)

    return data, np.array(features), (
        np.array(label) if mask is not None else None
    )

contours1, X1, y1 = getShapes(crop1, mask_crop1)
contours2, X2, y2 = getShapes(crop2, mask_crop2)

print("Shapes in image 1:", len(X1))
print("Training labels:", np.unique(y1, return_counts=True))
print("CandShapesidates in image 2:", len(X2))
print("Testing labels:", np.unique(y2, return_counts=True))

# 4. Train on image 1 and evaluate on image 2
# if len(np.unique(y1)) < 2:
#     raise ValueError(
#         "Training data has only one class. "
#         "Verify the mask colors and candidate labels."
#     )

#Implement random forest classifier
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

model.fit(X1, y1)
pred2 = model.predict(X2)

print("\nEvaluation on image 2:")
print(classification_report(
    y2, pred2,
    labels=[0, 1],
    target_names=['Background candidate', 'Barnacle candidate'],
    zero_division=0
))

# 5. Train on both labeled images for the unseen image
X_train = np.vstack([X1, X2])
y_train = np.concatenate([y1, y2])

if len(np.unique(y_train)) < 2:
    raise ValueError("Combined training data needs both classes.")

model.fit(X_train, y_train)



# Extract unseen-image candidates
contours_unseen, X_unseen, _ = getShapes(newImage)
pred_unseen = model.predict(X_unseen)

# 6. Count predicted barnacle candidates
barnacle_contours = [
    contour for contour, label in zip(contours_unseen, pred_unseen)
    if label == 1
]

print("\nPredicted barnacle candidates:", len(barnacle_contours))

# 7. Visualize detections
result = cropNew.copy()
cv.drawContours(result, barnacle_contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title(f"ML-predicted candidates: {len(barnacle_contours)}")
plt.axis("off")
plt.show()
