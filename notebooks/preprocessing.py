"""Image preprocessing shared by the face model (notebook 2) and the fusion demo (notebook 3)."""

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = (48, 48)
GABOR_ANGLES = (0, 30, 60, 90, 120, 150)


def load_face(path):
    """Load an image as a 48x48 grayscale uint8 array."""
    with Image.open(path) as img:
        return np.array(img.convert("L").resize(IMAGE_SIZE, resample=Image.LANCZOS))


def gabor_kernel(size=11, sigma=1.5, gamma=1.2, lambd=3, psi=0, angle=0):
    """Real Gabor kernel, normalised so that its absolute values sum to 1."""
    d = size // 2
    y, x = np.mgrid[-d : d + 1, -d : d + 1].astype(np.float32)
    theta = np.deg2rad(angle)
    xr = np.cos(theta) * x + np.sin(theta) * y
    yr = -np.sin(theta) * x + np.cos(theta) * y
    kernel = np.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * np.cos(2 * np.pi * xr / lambd + psi)
    return (kernel / np.abs(kernel).sum()).astype(np.float32)


KERNELS = [gabor_kernel(angle=a) for a in GABOR_ANGLES]


def gabor_features(img):
    """Sum of the responses of 6 Gabor filters (0-150 degrees), rescaled to 0-255.

    Each response is clipped to 0-255 before summing. Edges are padded by replicating
    the border pixels.
    """
    out = np.zeros(img.shape, dtype=np.float32)
    for kernel in KERNELS:
        response = cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        out += np.clip(response, 0, 255).astype(np.uint8)
    return (out / max(out.max(), 1) * 255).astype(np.uint8)


def to_model_input(images):
    """Gabor-filter a batch of 48x48 grayscale images and scale them to [0, 1] with a channel axis."""
    return np.stack([gabor_features(img) for img in images])[..., np.newaxis] / 255.0
