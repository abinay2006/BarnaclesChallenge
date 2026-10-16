
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# --------------------------------------------------
# 1. LOAD IMAGES AND MASKS
# --------------------------------------------------

img1 = cv.imread("images/img1.png")
img2 = cv.imread("images/img2.png")

mask1 = cv.imread("images/mask1.png")
mask2 = cv.imread("images/mask2.png")

newImage = cv.imread("images/unseen_img1.png")

# Check that files loaded successfully BEFORE processing them
files = {
    "img1": img1,
    "img2": img2,
    "mask1": mask1,
    "mask2": mask2,
    "unseen_img1": newImage
}

for name, file in files.items():
    if file is None:
        raise FileNotFoundError(
            f"Could not load {name}. Check the file path."
        )

# Convert masks to grayscale
mask1 = cv.cvtColor(mask1, cv.COLOR_BGR2GRAY)
mask2 = cv.cvtColor(mask2, cv.COLOR_BGR2GRAY)

print("Original image 1:", img1.shape)
print("Mask 1:", mask1.shape)
print("Original image 2:", img2.shape)
print("Mask 2:", mask2.shape)


# --------------------------------------------------
# 2. FIND THE ANNOTATED REGION
# --------------------------------------------------

def prepare_annotated_region(img, mask, title):
    outline = (mask < 200).astype(np.uint8) #determine if there is a barnacle based on pixel brightness since outline is darker

    # Find all nonzero annotation pixels
    points = cv.findNonZero(outline) #detect possibe barnacle outlines

    if points is None:
        raise cv.error("No barnacle regions found!")

    # find the rectangle around all the outline possible pixels from mask onto the real one
    x, y, w, h = cv.boundingRect(points)


    #make sure not out of bounds
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)

    # Crop the original image and mask with identical coordinates so pixels are't off later on in training
    image_crop = img[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    # show
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cv.imshow((image_crop, cv.COLOR_BGR2RGB))

    return image_crop, mask_crop


crop1, maskCrop1 = prepare_annotated_region(
    img1, mask1, "Image 1"
)

crop2, maskCrop2 = prepare_annotated_region(
    img2, mask2, "Image 2"
)


# --------------------------------------------------
# 3. DETECT CANDIDATE SHAPES AND EXTRACT FEATURES
# --------------------------------------------------

# 3. Generate contours of the images and extract features
def getShapes(img, mask=None): #the two images have the mask to detect barnacles but the new images don't
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 100, 200) #get the edges using Canny

    contours, hierarchy = cv.findContours( #trace boundaries using the edges to find shapes
        edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
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

        # Ignore tiny contours and very large regions so doesnt double count
        if area < 30 or area > img.shape[0] * img.shape[1] * 0.1:
            continue

        x, y, w, h = cv.boundingRect(contour) #make rectangle so easier to capture

        # Features
        oneShape = img[y:y+h, x:x+w] #color
        grayPart = gray[y:y+h, x:x+w] #gray
        edgePart = edges[y:y+h, x:x+w] #edge or not

        perimeter = cv.arcLength(contour, True)

        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2) #look at how close to a circle

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
            outlineRegion = outline[y:y+h, x:x+w]
            overlap = np.mean(outlineRegion > 0)
            label.append(1 if overlap > 0.02 else 0)

    return data, np.array(features), (
        np.array(label) if mask is not None else None
    )

# Generate candidate shapes and labels
contours1, X1, y1 = getShapes(crop1, maskCrop1)
contours2, X2, y2 = getShapes(crop2, maskCrop2)

# --------------------------------------------------
# 5. TRAIN ON IMAGE 1 AND EVALUATE ON IMAGE 2
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

model.fit(X1, y1)

pred2 = model.predict(X2)

print("\nEvaluation on image 2:")
print(
    classification_report(
        y2,
        pred2,
        labels=[0, 1],
        target_names=[
            "Background candidate",
            "Barnacle candidate"
        ],
        zero_division=0
    )
)


# --------------------------------------------------
# 6. TRAIN FINAL MODEL USING BOTH LABELED IMAGES
# --------------------------------------------------

X_train = np.vstack([X1, X2])
y_train = np.concatenate([y1, y2])

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. PROCESS THE UNSEEN IMAGE
# --------------------------------------------------

# At this stage, we have no mask for the unseen image.
# Detect candidates in the relevant region before predicting.

def select_region(img, title):

    x, y, w, h = cv.selectROI(
        title,
        img,
        fromCenter=False
    )

    cv.destroyWindow(title)

    if w == 0 or h == 0:
        raise ValueError("No region selected.")

    return img[y:y+h, x:x+w]


cropNew = select_region(
    newImage,
    "Select the relevant inner region"
)

contoursNew, featuresNew, _ = getShapes(cropNew)

if len(featuresNew) == 0:
    raise ValueError(
        "No candidate shapes found in the unseen image."
    )

predNew = model.predict(featuresNew)

barnacleContours = [
    contour
    for contour, label in zip(contoursNew, predNew)
    if label == 1
]

print(
    "\nPredicted barnacle candidates:",
    len(barnacleContours)
)


# --------------------------------------------------
# 8. DISPLAY PREDICTIONS
# --------------------------------------------------

result = cropNew.copy()

cv.drawContours(
    result,
    barnacleContours,
    -1,
    (0, 255, 0),
    2
)

plt.figure(figsize=(12, 7))

plt.imshow(
    cv.cvtColor(result, cv.COLOR_BGR2RGB)
)

plt.title(
    f"Predicted barnacle candidates: {len(barnacleContours)}"
)

plt.axis("off")
plt.tight_layout()
plt.show()
