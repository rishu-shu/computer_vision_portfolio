# Day 5 (Basics): Mouse Drawing App using OpenCV
# Draw using left mouse button
# Press 'c' to clear screen
# Press 'q' to quit

import cv2
import numpy as np
from pathlib import Path

# Create blank canvas
img = np.zeros((500, 500, 3), np.uint8)

drawing = False  # True when mouse is pressed

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Create window and bind mouse
cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Canvas", draw)

# Prepare outputs directory (script-relative)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUTPUT_DIR / "drawing.png"

print("Instructions: draw with left mouse button. Press 's' to save, 'c' to clear, 'q' to quit (saves on quit).")

while True:
    cv2.imshow("Canvas", img)

    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to clear
    if key == ord('c'):
        img[:] = 0

    # Save drawing
    elif key == ord('s'):
        ok = cv2.imwrite(str(OUT_FILE), img)
        print(f"Saved drawing to: {OUT_FILE} -> {ok}")

    # Press 'q' to quit (save automatically)
    elif key == ord('q'):
        ok = cv2.imwrite(str(OUT_FILE), img)
        print(f"Saved drawing to: {OUT_FILE} -> {ok}")
        break

cv2.destroyAllWindows()
