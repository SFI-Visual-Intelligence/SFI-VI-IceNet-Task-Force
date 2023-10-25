# To save a nice map highlighting Hudson Bay

import numpy as np
import matplotlib.pyplot as plt
import cv2

LAND_MASK_PATH = "./icenet/experimental/masks/land_mask.npy"

def createHudsonBaySystemImage():
    """
    Create an image highlighting the Hudson Bay System (HBS) region.
    """
    path = config.REGION_MASK_PATH
    region_mask = np.load(path)
    image = np.zeros(region_mask.shape + (3,), dtype=int)

    red = (150, 0, 0)
    green = (0, 150, 0)
    blue = (6, 66, 150)

    image[np.where(region_mask == 5)] = red
    image[np.where(region_mask == 12)] = green
    image[np.where(region_mask != 12)] = blue

    return image


def createRegionImage(region_mask=np.load(config.REGION_MASK_PATH), region=5):
    """
    Create an image highlighting a region.
    """
    image = np.zeros(region_mask.shape + (3,), dtype=int)

    red = (150, 0, 0)
    green = (0, 150, 0)
    blue = (6, 66, 150)

    image[np.where(region_mask == 12)] = green
    image[np.where(region_mask != 12)] = blue
    image[np.where(region_mask == region)] = red

    return image


if __name__ == "__main__":
    image = createHudsonBaySystemImage()
    plt.imshow(image)
    plt.suptitle("The Hudson Bay System (HBS)", size=12)
    plt.title("Hudson Bay (incl. James Bay), Foxe Basin and Hudson Strait", size=8)
    plt.axis("off")
    plt.savefig("./figures/HudsonBaySystem.jpg", dpi=1200)
    plt.show()


def createLandEdgeImage(land_mask=np.load(LAND_MASK_PATH), method="sobel"):
    """
    Create an image of land edges.
    """
    
    if method == "sobel":
        land_edge = cv2.Sobel(
            (land_mask[:, :] * 1).astype(float), cv2.CV_64F, 1, 0, ksize=5
        )
    elif method == "laplacian":
        land_edge = cv2.Laplacian((land_mask[:, :] * 1).astype(float), cv2.CV_64F)
    else:
        raise ValueError("method must be either 'sobel' or 'laplacian'")
    return land_edge


if __name__ == "__main__":
    land_edge = createLandEdgeImage()
    plt.imshow(land_edge)
    plt.axis("off")
    plt.savefig("./figures/land_edge.jpg", dpi=1200)
    plt.show()


if __name__ == "__main__":
    for i in [2, 3, 4, 6, 7, 8, 9, 10, 11, 13]:    
        image = createRegionImage(region=i)
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(f"./figures/land_mask_{i}.jpg", dpi=1200)
        plt.close()
