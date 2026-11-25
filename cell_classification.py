
import os
import csv
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from models import create_cell_classifier

# -----------------------------------
# 1. Configuration
# -----------------------------------

# Placeholder paths
IMAGE_PATH = "testis.png"
MASK_PATH = "cell_masks.npy"

MODEL_PATH = "checkpoints/best_cell_classifier.pth"

OUTPUT_CSV = "cell_predictions.csv"
OUTPUT_IMAGE = "classified_cells.png"

BATCH_SIZE = 32
PADDING = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------
# 2. Load original image and masks
# -----------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

masks = np.load(MASK_PATH, allow_pickle=False)

if masks.ndim != 2:
    raise ValueError("Expected a 2D Cellpose mask.")

if masks.shape != image.shape[:2]:
    raise ValueError(
        "Image and segmentation mask dimensions do not match."
    )

print("Image shape:", image.shape)
print("Detected cells:", np.count_nonzero(np.unique(masks)))

# -----------------------------------
# 3. Load trained ResNet-50
# -----------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

class_names = checkpoint["class_names"]

model = create_cell_classifier(
    pretrained=False
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

# -----------------------------------
# 4. Image preprocessing
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
# 5. Extract cell crops
# -----------------------------------

cells = []
cell_ids = []
locations = []

height, width = masks.shape

for cell_id in np.unique(masks):
    if cell_id == 0:
        continue

    ys, xs = np.where(masks == cell_id)

    if len(xs) == 0:
        continue

    x_min = max(0, xs.min() - PADDING)
    x_max = min(width, xs.max() + PADDING + 1)
    y_min = max(0, ys.min() - PADDING)
    y_max = min(height, ys.max() + PADDING + 1)

    crop = image[y_min:y_max, x_min:x_max]

    cells.append(Image.fromarray(crop))
    cell_ids.append(int(cell_id))
    locations.append((
        float(xs.mean()),
        float(ys.mean())
    ))

# -----------------------------------
# 6. Classify all extracted cells
# -----------------------------------

results = []

with torch.no_grad():
    for start in range(0, len(cells), BATCH_SIZE):

        batch_cells = cells[start:start + BATCH_SIZE]

        batch = torch.stack([
            transform(cell) for cell in batch_cells
        ]).to(device)

        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        for i in range(len(batch_cells)):
            idx = start + i

            results.append({
                "cell_id": cell_ids[idx],
                "x": locations[idx][0],
                "y": locations[idx][1],
                "class_index": predictions[i].item(),
                "cell_type": class_names[predictions[i].item()],
                "confidence": confidence[i].item()
            })

# -----------------------------------
# 7. Save predictions to CSV
# -----------------------------------

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "cell_id",
            "x",
            "y",
            "class_index",
            "cell_type",
            "confidence"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print(f"Saved predictions to {OUTPUT_CSV}")

# -----------------------------------
# 8. Visualize predictions
# -----------------------------------

fig, ax = plt.subplots(figsize=(14, 14))

ax.imshow(image)

for result in results:
    x = result["x"]
    y = result["y"]

    ax.text(
        x, y,
        str(result["class_index"]),
        fontsize=6,
        color="yellow",
        ha="center",
        va="center"
    )

ax.set_title("Cell Classification Results")
ax.axis("off")

plt.savefig(
    OUTPUT_IMAGE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved visualization to {OUTPUT_IMAGE}")