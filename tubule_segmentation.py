import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp

# -----------------------------------
# 1. Initialize DeepLabV3+
# -----------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2
).to(device)

model.eval()

# -----------------------------------
# 2. Load histology image
# -----------------------------------

image = cv2.imread("testis.png")

if image is None:
    raise FileNotFoundError("Could not find testis.png")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original_height, original_width = image.shape[:2]

# -----------------------------------
# 3. Preprocess image
# -----------------------------------

# Resize for model input
resized = cv2.resize(image, (512, 512))

# Normalize using ImageNet statistics
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

normalized = resized.astype(np.float32) / 255.0
normalized = (normalized - mean) / std

# Convert to PyTorch tensor: [1, 3, 512, 512]
tensor = torch.from_numpy(
    normalized.transpose(2, 0, 1)
).unsqueeze(0).float().to(device)

# -----------------------------------
# 4. Run segmentation
# -----------------------------------

with torch.no_grad():
    logits = model(tensor)

    probabilities = torch.softmax(logits, dim=1)

    mask = torch.argmax(
        probabilities,
        dim=1
    )[0].cpu().numpy().astype(np.uint8)

# Resize mask to original image dimensions
mask = cv2.resize(
    mask,
    (original_width, original_height),
    interpolation=cv2.INTER_NEAREST
)

# -----------------------------------
# 5. Visualize results
# -----------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Histology Image")

# Segmentation mask
axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Tubule Segmentation Mask")

# Overlay on original image
overlay = image.copy()

overlay[mask == 1] = (
    0.5 * overlay[mask == 1] +
    0.5 * np.array([255, 0, 0])
).astype(np.uint8)

axes[2].imshow(overlay)
axes[2].set_title("Tubule Segmentation Overlay")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("tubule_segmentation.png", dpi=300)
plt.show()

# Save numerical mask
np.save("tubule_masks.npy", mask)