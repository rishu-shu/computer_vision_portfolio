import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Script directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Allow overrides via environment variables
env_input = os.environ.get("MY_INPUT_PATH")
env_output = os.environ.get("MY_OUTPUT_DIR")

if env_input:
    INPUT_PATH = Path(env_input)
else:
    INPUT_PATH = SCRIPT_DIR / "test.jpg"

if env_output:
    OUTPUT_DIR = Path(env_output)
else:
    OUTPUT_DIR = SCRIPT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Script dir: {SCRIPT_DIR}")
print(f"Current working dir: {Path.cwd()}")
print(f"Using input path: {INPUT_PATH}")
print(f"Using output dir: {OUTPUT_DIR}")

# LOAD IMAGE
img = cv2.imread(str(INPUT_PATH))
if img is None:
    print("Error: Image not found. Please check the path.")
    print(f"Tried path: {INPUT_PATH}")
    sys.exit(1)

def safe_imshow(name, image):
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
    except Exception as e:
        print(f"Warning: can't show window '{name}': {e}")

safe_imshow("Original", img)

# BLUR
blurred=cv2.blur(img,(15,15))
safe_imshow("Blurred", blurred)

#GAUSSIAN BLUR
gaussian_blurred=cv2.GaussianBlur(img,(15,15),0)
safe_imshow("Gaussian Blurred", gaussian_blurred)

#MEDIAN BLUR
median_blurred=cv2.medianBlur(img,15)
safe_imshow("Median Blurred", median_blurred)

#EDGE DETECTION
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,100,200)
safe_imshow("Edges", edges)

# KERNEL
kernel=np.ones((2,2),np.uint8)

#DILATION (expands white areas)
dilated=cv2.dilate(edges,kernel,iterations=2)
safe_imshow("Dilated", dilated)

#EROSION (contracts white areas)
eroded=cv2.erode(edges,kernel,iterations=1)
safe_imshow("Eroded", eroded)

#THRESHOLDING
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
safe_imshow("Thresholded", thresh)

#OPENING (removes noise)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
safe_imshow("Opening", opening)

#CLOSING (fills holes)
closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
safe_imshow("Closing", closing)

# SHARPENING
kernel_=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpened=cv2.filter2D(img,-1,kernel_)
safe_imshow("Sharpened", sharpened)

# SAVE IMAGES
out_blurred = OUTPUT_DIR / "blurred.jpg"
out_gaussian = OUTPUT_DIR / "gaussian_blurred.jpg"
out_median = OUTPUT_DIR / "median_blurred.jpg"
out_edges = OUTPUT_DIR / "edges.jpg"
out_dilated = OUTPUT_DIR / "dilated.jpg"
out_eroded = OUTPUT_DIR / "eroded.jpg"
out_thresh = OUTPUT_DIR / "thresholded.jpg"
out_opening = OUTPUT_DIR / "opening.jpg"
out_closing = OUTPUT_DIR / "closing.jpg"
out_sharpened = OUTPUT_DIR / "sharpened.jpg"

ok1 = cv2.imwrite(str(out_blurred), blurred)
ok2 = cv2.imwrite(str(out_gaussian), gaussian_blurred)
ok3 = cv2.imwrite(str(out_median), median_blurred)
ok4 = cv2.imwrite(str(out_edges), edges)
ok5 = cv2.imwrite(str(out_dilated), dilated)
ok6 = cv2.imwrite(str(out_eroded), eroded)
ok7 = cv2.imwrite(str(out_thresh), thresh)
ok8 = cv2.imwrite(str(out_opening), opening)
ok9 = cv2.imwrite(str(out_closing), closing)
ok10 = cv2.imwrite(str(out_sharpened), sharpened)

print("Saved files:")
print(f" - {out_blurred}: {ok1}")
print(f" - {out_gaussian}: {ok2}")
print(f" - {out_median}: {ok3}")
print(f" - {out_edges}: {ok4}")
print(f" - {out_dilated}: {ok5}")
print(f" - {out_eroded}: {ok6}")
print(f" - {out_thresh}: {ok7}")
print(f" - {out_opening}: {ok8}")
print(f" - {out_closing}: {ok9}")
print(f" - {out_sharpened}: {ok10}")
try:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except Exception:
    pass
