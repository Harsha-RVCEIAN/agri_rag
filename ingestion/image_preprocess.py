# ingestion/image_preprocess.py

import cv2
import numpy as np
import math

# =========================
# CONFIG
# =========================

TARGET_DPI = 300
MAX_DESKEW_ANGLE = 5.0           # degrees
GAUSSIAN_BLUR_KERNEL = (3, 3)

MIN_IMAGE_AREA = 300 * 300       # 🔑 tiny images are junk
MAX_SCALE_FACTOR = 4.0           # 🔑 prevent hallucinated resolution
MIN_EDGE_DENSITY = 0.002         # 🔑 detect blank / low-text images


# =========================
# DPI NORMALIZATION
# =========================

def resize_to_target_dpi(image: np.ndarray, current_dpi: int = 72) -> np.ndarray:
    if current_dpi <= 0 or current_dpi == TARGET_DPI:
        return image

    scale = TARGET_DPI / current_dpi
    scale = min(scale, MAX_SCALE_FACTOR)

    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# =========================
# NOISE REMOVAL
# =========================

def remove_noise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=25)


# =========================
# CONTRAST ENHANCEMENT
# =========================

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=1.8,            # 🔑 lower = safer
        tileGridSize=(8, 8)
    )
    return clahe.apply(gray)


# =========================
# BINARIZATION
# =========================

def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3
    )


# =========================
# DESKEWING
# =========================

def estimate_skew_angle(binary: np.ndarray) -> float:
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0.0

    angles = []
    for i in range(min(len(lines), 15)):
        rho, theta = lines[i][0]
        angle = (theta - np.pi / 2) * 180 / math.pi
        angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def deskew(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.2 or abs(angle) > MAX_DESKEW_ANGLE:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# =========================
# QUALITY GUARDS
# =========================

def _edge_density(binary: np.ndarray) -> float:
    edges = cv2.Canny(binary, 50, 150)
    return edges.sum() / max(binary.size, 1)


# =========================
# MAIN PIPELINE
# =========================

def preprocess_image(image: np.ndarray, current_dpi: int = 72) -> np.ndarray:
    """
    Conservative OCR preprocessing.
    Improves clarity WITHOUT fabricating confidence.
    """

    # ---------- basic sanity ----------
    h, w = image.shape[:2]
    if h * w < MIN_IMAGE_AREA:
        return image  # 🔑 too small → don't fake it

    # 1️⃣ DPI normalization
    image = resize_to_target_dpi(image, current_dpi)

    # 2️⃣ Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Noise removal
    gray = remove_noise(gray)

    # 4️⃣ Contrast enhancement
    gray = enhance_contrast(gray)

    # 5️⃣ Binarization
    binary = adaptive_binarize(gray)

    # ---------- blank / low-text guard ----------
    if _edge_density(binary) < MIN_EDGE_DENSITY:
        return binary  # 🔑 no deskewing, no overprocessing

    # 6️⃣ Deskew
    angle = estimate_skew_angle(binary)
    binary = deskew(binary, angle)

    return binary
