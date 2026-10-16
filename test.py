
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load images and masks
img1 = cv.imread('images/img1.png')
img2 = cv.imread('images/img2.png')
mask1 = cv.imread('images/mask1.png', cv.IMREAD_GRAYSCALE)
mask2 = cv.imread('images/mask2.png', cv.IMREAD_GRAYSCALE)
unseen = cv.imread('images/unseen_img1.png')

if any(x is None for x in [img1, img2, mask1, mask2, unseen]):
    raise FileNotFoundError("Check your image paths.")

# 2. Select the inner green square in each image
def crop_image(img, name):
    x, y, w, h = cv.selectROI(
        name, img, showCrosshair=True, fromCenter=False
    )
    cv.destroyWindow(name)

    if w == 0 or h == 0:
        raise ValueError("You must select the inner square.")

    return img[y:y+h, x:x+w], (x, y, w, h)

crop1, roi1 = crop_image(img1, "Select square in img1")
crop2, roi2 = crop_image(img2, "Select square in img2")
crop_unseen, roi_unseen = crop_image(unseen, "Select square in unseen")

def crop_mask(mask, roi):
    x, y, w, h = roi
    return mask[y:y+h, x:x+w]

mask_crop1 = crop_mask(mask1, roi1)
mask_crop2 = crop_mask(mask2, roi2)

# 3. Generate candidate contours and extract features
def get_candidates(img, mask=None):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 125, 175)

    contours, _ = cv.findContours(
        edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
    )

    # Dilate annotated outlines so nearby candidates can match them.
    if mask is not None:
        # Based on the observed mask values (66, 80, 255):
        # assumes darker pixels mark annotated outlines.
        outline = (mask < 200).astype(np.uint8)
        outline = cv.dilate(
            outline, np.ones((7, 7), np.uint8)
        )

    candidates = []
    features = []
    labels = []

    for contour in contours:
        area = cv.contourArea(contour)

        # Ignore tiny contours and very large regions
        if area < 30 or area > img.shape[0] * img.shape[1] * 0.1:
            continue

        x, y, w, h = cv.boundingRect(contour)

        if w < 2 or h < 2:
            continue

        # Features: size, shape, color, brightness, and edge density
        patch = img[y:y+h, x:x+w]
        gray_patch = gray[y:y+h, x:x+w]
        edge_patch = edges[y:y+h, x:x+w]

        perimeter = cv.arcLength(contour, True)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)

        features.append([
            area,
            w / h,
            circularity,
            *np.mean(patch, axis=(0, 1)),
            *np.std(patch, axis=(0, 1)),
            np.mean(gray_patch),
            np.std(gray_patch),
            np.mean(edge_patch > 0)
        ])

        candidates.append(contour)

        if mask is not None:
            # Approximate label based on overlap with annotated outlines
            outline_region = outline[y:y+h, x:x+w]
            overlap = np.mean(outline_region > 0)
            labels.append(1 if overlap > 0.02 else 0)

    return candidates, np.array(features), (
        np.array(labels) if mask is not None else None
    )

contours1, X1, y1 = get_candidates(crop1, mask_crop1)
contours2, X2, y2 = get_candidates(crop2, mask_crop2)

print("Candidates in image 1:", len(X1))
print("Training labels:", np.unique(y1, return_counts=True))
print("Candidates in image 2:", len(X2))
print("Testing labels:", np.unique(y2, return_counts=True))

# 4. Train on image 1 and evaluate on image 2
if len(np.unique(y1)) < 2:
    raise ValueError(
        "Training data has only one class. "
        "Verify the mask colors and candidate labels."
    )

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
contours_unseen, X_unseen, _ = get_candidates(crop_unseen)
pred_unseen = model.predict(X_unseen)

# 6. Count predicted barnacle candidates
barnacle_contours = [
    contour for contour, label in zip(contours_unseen, pred_unseen)
    if label == 1
]

print("\nPredicted barnacle candidates:", len(barnacle_contours))

# 7. Visualize detections
result = crop_unseen.copy()
cv.drawContours(result, barnacle_contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(12, 6))
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title(f"ML-predicted candidates: {len(barnacle_contours)}")
plt.axis("off")
plt.show()
