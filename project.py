
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load images
img1 = cv.imread('images/img1.png')
img2 = cv.imread('images/img2.png')
mask1 = cv.imread('images/mask1.png', cv.IMREAD_GRAYSCALE)
mask2 = cv.imread('images/mask2.png', cv.IMREAD_GRAYSCALE)
unseen = cv.imread('images/unseen_img1.png')

if any(x is None for x in [img1, img2, mask1, mask2, unseen]):
    raise FileNotFoundError("Check your image paths and filenames.")

# Make sure each mask matches its image
if img1.shape[:2] != mask1.shape or img2.shape[:2] != mask2.shape:
    raise ValueError("Each mask must have the same dimensions as its image.")

# 2. Generate approximate labels from mask outlines

# Check what is actually inside each mask
for name, mask in [('mask1', mask1), ('mask2', mask2)]:
    print(f"\n{name}")
    print("Shape:", mask.shape)
    print("Min/max:", mask.min(), mask.max())
    print("Unique values:", np.unique(mask)[:20])
    print("Pixel counts:", np.unique(mask, return_counts=True))

# Create approximate labels

def make_labels(mask):
    # Initial hypothesis: darker pixels mark the barnacle outlines
    outline = (mask == 66).astype(np.uint8)

    print("Outline pixels:", np.count_nonzero(outline))
    print("Total pixels:", outline.size)

    positive = cv.dilate(
        outline, np.ones((7, 7), np.uint8)
    ).astype(bool)

    wide = cv.dilate(
        outline, np.ones((21, 21), np.uint8)
    ).astype(bool)

    labels = np.full(mask.shape, -1, dtype=np.int8)
    labels[wide == 0] = 0
    labels[positive] = 1

    print("Generated labels:", np.unique(labels, return_counts=True))
    return labels

# 3. Prepare training data
def get_data(img, labels):
    valid = labels >= 0

    X = img[valid]
    y = labels[valid]

    # Sample pixels to keep training quick
    rng = np.random.default_rng(42)

    if len(y) > 10000:
        idx = rng.choice(len(y), 10000, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y
labels1 = make_labels(mask1)
labels2 = make_labels(mask2)

X_train, y_train = get_data(img1, labels1)
X_test, y_test = get_data(img2, labels2)

# Check that training data contains both classes
if len(np.unique(y_train)) < 2:
    raise ValueError(
        "Training data contains only one class. "
        "Check mask1.png and the generated labels."
    )

if len(np.unique(y_test)) < 2:
    print("Warning: Test data contains only one class.")
    print("Metrics will not fully evaluate both classes.")

# 4. Train the ML classifier
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# 5. Evaluate on image 2
predictions = model.predict(X_test)

print("\nClassification results:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=['Background', 'Barnacle'],
    zero_division=0
))

# 6. Predict barnacle-like pixels in the unseen image
pixels = unseen.reshape(-1, 3)
predictions_unseen = model.predict(pixels)
prediction_map = predictions_unseen.reshape(unseen.shape[:2])

# 7. Display results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(unseen, cv.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(prediction_map, cmap='gray', vmin=0, vmax=1)
plt.title('Predicted Barnacle Pixels')
plt.axis('off')

plt.tight_layout()
plt.show()
