
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
    overlay = image_crop.copy()

    annotation_pixels = mask_crop < 200
    overlay[annotation_pixels] = (0, 0, 255)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        cv.cvtColor(image_crop, cv.COLOR_BGR2RGB)
    )
    axes[0].set_title("Image crop")

    axes[1].imshow(mask_crop, cmap="gray")
    axes[1].set_title("Mask annotations")

    axes[2].imshow(
        cv.cvtColor(overlay, cv.COLOR_BGR2RGB)
    )
    axes[2].set_title("Red = annotation pixels")

    for ax in axes:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

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

def getShapes(img, mask=None):

    # Convert to grayscale for edge detection
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv.Canny(gray, 100, 200)

    # Find contours around edges
    contours, hierarchy = cv.findContours(
        edges,
        cv.RETR_LIST,
        cv.CHAIN_APPROX_SIMPLE
    )

    # Prepare annotation pixels if a mask is provided
    if mask is not None:

        annotation = (mask < 200).astype(np.uint8)

        # Thicken thin annotation outlines slightly
        kernel = np.ones((7, 7), np.uint8)
        annotation = cv.dilate(annotation, kernel)

    data = []
    features = []
    labels = []

    for contour in contours:

        area = cv.contourArea(contour)

        # Ignore very small and extremely large contours
        if area < 30:
            continue

        if area > img.shape[0] * img.shape[1] * 0.1:
            continue

        # Get the rectangle surrounding this contour
        x, y, w, h = cv.boundingRect(contour)

        # Extract the corresponding image regions
        colorPart = img[y:y+h, x:x+w]
        grayPart = gray[y:y+h, x:x+w]
        edgePart = edges[y:y+h, x:x+w]

        # Calculate shape features
        perimeter = cv.arcLength(contour, True)

        circularity = 0

        if perimeter > 0:
            circularity = (
                4 * np.pi * area / (perimeter ** 2)
            )

        # Calculate color and edge features
        meanColor = np.mean(colorPart, axis=(0, 1))
        grayMean = np.mean(grayPart)
        edgeDensity = np.mean(edgePart > 0)

        # Feature vector:
        # area, circularity, B, G, R, grayscale mean, edge density
        features.append([
            area,
            circularity,
            *meanColor,
            grayMean,
            edgeDensity
        ])

        data.append(contour)

        # Generate an approximate label from the mask
        if mask is not None:

            annotationPart = annotation[y:y+h, x:x+w]

            # Fraction of the candidate's bounding box that
            # overlaps the dilated annotation outlines
            overlap = np.mean(annotationPart > 0)

            # Approximate labels:
            # 1 = overlaps annotation outlines
            # 0 = does not overlap enough
            labels.append(1 if overlap > 0.02 else 0)

    return (
        data,
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=np.int32) if mask is not None else None
    )


# Generate candidate shapes and labels
contours1, X1, y1 = getShapes(crop1, maskCrop1)
contours2, X2, y2 = getShapes(crop2, maskCrop2)

print("\nCandidate counts")
print("Image 1:", len(contours1))
print("Image 2:", len(contours2))

print("\nImage 1 label distribution:")
print(np.unique(y1, return_counts=True))

print("\nImage 2 label distribution:")
print(np.unique(y2, return_counts=True))


# --------------------------------------------------
# 4. CHECK THAT TRAINING DATA IS USABLE
# --------------------------------------------------

if len(X1) == 0 or len(X2) == 0:
    raise ValueError(
        "No candidate shapes were detected. "
        "Try adjusting the Canny thresholds or area filters."
    )

if len(np.unique(y1)) < 2:
    raise ValueError(
        "Image 1 contains only one generated label class. "
        "The classifier needs positive and negative examples."
    )

if len(np.unique(y2)) < 2:
    print(
        "\nWARNING: Image 2 contains only one generated label class. "
        "The evaluation may be misleading."
    )


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
