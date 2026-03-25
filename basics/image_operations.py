
# OpenCV Image Processing Script
# File handling improved using pathlib to make paths independent of working directory.
# GitHub Copilot was used to assist in implementing file handling logic.

import cv2
import sys
from pathlib import Path

# Make paths relative to this script so behavior doesn't depend on CWD
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / "test.jpg"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Script dir: {SCRIPT_DIR}")
print(f"Current working dir: {Path.cwd()}")
print(f"Looking for input image at: {INPUT_PATH}")

# LOAD IMAGE
img = cv2.imread(str(INPUT_PATH))
if img is None:
    print(f"ERROR: Image not found. Tried: {INPUT_PATH}")
    sys.exit(1)

# Safe imshow (won't crash in headless environments)
def safe_imshow(name, image):
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
    except Exception as e:
        print(f"Warning: can't show window '{name}': {e}")

# ORIGINAL
safe_imshow("Original", img)

# IMAGE INFO
print("Shape (H, W, C):", img.shape)
print("Data type:", img.dtype)

# GRAYSCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
safe_imshow("Grayscale", gray)

# RESIZE
resized = cv2.resize(img, (1000, 1700))
safe_imshow("Resized", resized)

# SAFE CROP
h, w = img.shape[:2]
cropped = None
if h > 500 and w > 1000:
    cropped = img[10:500, 30:1000]
    safe_imshow("Cropped", cropped)

# ROTATE
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated = cv2.warpAffine(img, matrix, (w, h))
safe_imshow("Rotated", rotated)

# SAVE IMAGES (use script-relative outputs directory)
out_gray = OUTPUT_DIR / "gray.jpg"
out_resized = OUTPUT_DIR / "resized.jpg"
out_cropped = OUTPUT_DIR / "cropped.jpg"
out_rotated = OUTPUT_DIR / "rotated.jpg"

ok1 = cv2.imwrite(str(out_gray), gray)
ok2 = cv2.imwrite(str(out_resized), resized)
ok3 = cv2.imwrite(str(out_cropped), cropped) if cropped is not None else False
ok4 = cv2.imwrite(str(out_rotated), rotated)

print("Saved files:")
print(f" - {out_gray}: {ok1}")
print(f" - {out_resized}: {ok2}")
print(f" - {out_cropped}: {ok3}")
print(f" - {out_rotated}: {ok4}")

try:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass