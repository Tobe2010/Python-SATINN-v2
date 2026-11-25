import torch
import numpy as np
import matplotlib.pyplot as plt

from cellpose import models, io, plot

# ----------------------------
# 1. Initialize Cellpose 2.0
# ----------------------------

use_gpu = torch.cuda.is_available()

model = models.Cellpose(
    gpu=use_gpu,
    model_type="cyto2"
)

# ----------------------------
# 2. Load an actual image
# ----------------------------

image_path = "testis.png"

image = io.imread(image_path)

print("Image shape:", image.shape)

# ----------------------------
# 3. Run cell segmentation
# ----------------------------

masks, flows, styles, diams = model.eval(
    image,
    diameter=None,
    channels=[0, 0]
)

print("Cells detected:", int(masks.max()))

# ----------------------------
# 4. Visualize the results
# ----------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Histology Image")

# Colored segmentation masks
axes[1].imshow(masks, cmap="nipy_spectral")
axes[1].set_title("Cell Segmentation Masks")

# Segmentation outlines on original image
outlines = plot.outlines_list(masks)

axes[2].imshow(image)

for outline in outlines:
    axes[2].plot(
        outline[:, 0],
        outline[:, 1],
        color="red",
        linewidth=0.7
    )

axes[2].set_title("Detected Cell Boundaries")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()