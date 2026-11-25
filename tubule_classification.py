
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

# -----------------------------------
# 1. Configuration
# -----------------------------------

IMAGE_PATH = "testis.png"
MASK_PATH = "tubule_masks.npy"
MODEL_PATH = "checkpoints/best_tubule_classifier.pth"

OUTPUT_CSV = "tubule_predictions.csv"
OUTPUT_IMAGE = "classified_tubules.png"
OUTPUT_MASK = "tubule_instance_masks.npy"

BATCH_SIZE = 16
PADDING = 15
MIN_AREA = 500

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------
# 2. Load original image and mask
# -----------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = np.load(MASK_PATH, allow_pickle=False)

if mask.shape != image.shape[:2]:
    raise ValueError(
        "Tubule mask and image dimensions do not match."
    )

# Convert semantic mask to binary
binary_mask = (mask == 1).astype(np.uint8)

# -----------------------------------
# 3. Identify individual tubules
# -----------------------------------

num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )
)

instance_masks = np.zeros(
    binary_mask.shape,
    dtype=np.int32
)

tubules = []
tubule_ids = []
locations = []
areas = []

next_id = 1

for label_id in range(1, num_labels):

    area = stats[label_id, cv2.CC_STAT_AREA]

    if area < MIN_AREA:
        continue

    x = stats[label_id, cv2.CC_STAT_LEFT]
    y = stats[label_id, cv2.CC_STAT_TOP]
    w = stats[label_id, cv2.CC_STAT_WIDTH]
    h = stats[label_id, cv2.CC_STAT_HEIGHT]

    x_min = max(0, x - PADDING)
    y_min = max(0, y - PADDING)
    x_max = min(image.shape[1], x + w + PADDING)
    y_max = min(image.shape[0], y + h + PADDING)

    crop = image[y_min:y_max, x_min:x_max]

    tubules.append(Image.fromarray(crop))
    tubule_ids.append(next_id)

    locations.append((
        float(centroids[label_id][0]),
        float(centroids[label_id][1])
    ))

    areas.append(int(area))

    instance_masks[labels == label_id] = next_id

    next_id += 1

print("Detected tubules:", len(tubules))

np.save(OUTPUT_MASK, instance_masks)

# -----------------------------------
# 4. Load trained ResNet-50
# -----------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model = resnet50(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    12
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

# -----------------------------------
# 5. Preprocess tubule crops
# -----------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------------
# 6. Classify each tubule
# -----------------------------------

results = []

with torch.no_grad():

    for start in range(0, len(tubules), BATCH_SIZE):

        batch_tubules = tubules[start:start + BATCH_SIZE]

        batch = torch.stack([
            transform(tubule)
            for tubule in batch_tubules
        ]).to(device)

        logits = model(batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predictions = (
            probabilities.max(dim=1)
        )

        for i in range(len(batch_tubules)):

            idx = start + i

            stage = predictions[i].item() + 1

            results.append({
                "tubule_id": tubule_ids[idx],
                "x": locations[idx][0],
                "y": locations[idx][1],
                "area_pixels": areas[idx],
                "stage": stage,
                "confidence": confidence[i].item()
            })

# -----------------------------------
# 7. Save predictions
# -----------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "tubule_id",
            "x",
            "y",
            "area_pixels",
            "stage",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("Saved:", OUTPUT_CSV)

# -----------------------------------
# 8. Visualize predictions
# -----------------------------------

fig, ax = plt.subplots(figsize=(14, 14))

ax.imshow(image)

for result in results:

    x = result["x"]
    y = result["y"]

    ax.text(
        x,
        y,
        f"Stage {result['stage']}",
        fontsize=9,
        color="yellow",
        ha="center",
        va="center",
        bbox={
            "facecolor": "black",
            "alpha": 0.5,
            "pad": 2
        }
    )

# Draw tubule boundaries
contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

ax.imshow(
    np.ma.masked_where(instance_masks == 0, instance_masks),
    cmap="nipy_spectral",
    alpha=0.15
)

ax.set_title("Tubule Stage Classification")
ax.axis("off")

plt.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved:", OUTPUT_IMAGE)