import numpy as np
import matplotlib.pyplot as plt

# load npz file as numpy array
heatmap = np.load("./icenet/experimental/results/spatial_heatmap_080323_16:41.npz")["arr_0"]

# load the land egde map
from icenet.experimental.config import LAND_MASK_PATH
land_mask = np.load(LAND_MASK_PATH)

from icenet.experimental.make_hbs_map import createLandEdgeImage

land_edge = createLandEdgeImage()

# plot all leadtimes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

vmin = 0 #np.min(heatmap)
vmax = np.max(heatmap)

for i, ax in enumerate(axs.flatten()):
    ax.imshow(heatmap[:, :, i]*(heatmap[:, :, i] < 0)*(-1), cmap="hot", vmin=vmin, vmax=vmax)
    ax.set_title(f"Leadtime {i+1}")
    ax.set_xticks([])
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()

plt.imshow(np.abs(heatmap.sum(axis=-1)), cmap="hot")
plt.show()

# heatmap and land edge
plt.imshow(heatmap.sum(axis=-1) != 0, cmap="gray")
plt.imshow(land_edge, cmap="gray", alpha=0.5)
plt.show()


fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(heatmap.sum(axis=-1) != 0, cmap="gray")
# plot land edge
axs[0, 0].imshow(land_edge, cmap="gray", alpha=0.5)
axs[0, 0].set_title("Receptive field")
axs[0, 0].axis("off")

# use a warm colormap for positive feature importance
axs[0, 1].imshow(heatmap.sum(axis=-1) > 0, cmap="Reds")
axs[0, 1].imshow(land_edge, cmap="gray", alpha=0.5)
axs[0, 1].set_title("Values > 0")
axs[0, 1].axis("off")

axs[0, 2].imshow(heatmap.sum(axis=-1) < 0, cmap="Blues")
axs[0, 2].imshow(land_edge, cmap="gray", alpha=0.5)
axs[0, 2].set_title("Values < 0")
axs[0, 2].axis("off")

axs[1, 0].imshow((heatmap.sum(axis=-1) > 0)*(heatmap.sum(axis=-1)), cmap="hot")
axs[1, 0].imshow(land_edge, cmap="gray", alpha=0.3)
axs[1, 0].set_title("Positive values")
axs[1, 0].axis("off")

axs[1, 1].imshow((heatmap.sum(axis=-1) < 0)*(-1*heatmap.sum(axis=-1)), cmap="hot")
axs[1, 1].imshow(land_edge, cmap="gray", alpha=0.3)
axs[1, 1].set_title("Negative values")
axs[1, 1].axis("off")

axs[1, 2].imshow(np.abs(heatmap.sum(axis=-1)), cmap="hot")
axs[1, 2].imshow(land_edge, cmap="gray", alpha=0.3)
axs[1, 2].set_title("Sum of absolute values")
axs[1, 2].axis("off")

plt.show()



# use a cold colormap
plt.imshow(np.abs(heatmap.sum(axis=-1))*land_mask, cmap="hot")
plt.imshow(land_edge, cmap="gray", alpha=0.3)
plt.axis("off")
plt.show()

plt.imshow(land_mask, cmap="gray")
plt.show()

# plot all leadtimes
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, ax in enumerate(axs.flatten()):
    ax.imshow(heatmap[:, :, i] != 0, cmap="hot")
    ax.set_title(f"Leadtime {i+1}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()