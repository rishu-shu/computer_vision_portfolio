import cv2
import numpy as np
import argparse
from pathlib import Path

# CLI: allow overriding output directory
parser = argparse.ArgumentParser(description="Draw shapes and save to output folder")
parser.add_argument('--output', '-o', help='Output directory for the saved image')
parser.add_argument('--show', action='store_true', help='Show GUI window (interactive)')
args = parser.parse_args()

# Base script dir
SCRIPT_DIR = Path(__file__).resolve().parent

# Determine output dir: CLI -> env MY_OUTPUT_DIR -> script-relative outputs
if args.output:
    OUTPUT_DIR = Path(args.output)
else:
    OUTPUT_DIR = SCRIPT_DIR / 'outputs'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create a blank image (black canvas)
img = np.zeros((500, 500, 3), dtype=np.uint8)

#DRAWING
# LINE
cv2.line(img, (50, 50), (450, 50), (255, 0, 0), 3)

# RECTANGLE
cv2.rectangle(img, (50, 100), (200, 250), (0, 255, 0), 3)

# Filled rectangle
cv2.rectangle(img, (250, 100), (450, 250), (0, 255, 0), -1)

# CIRCLE
cv2.circle(img, (150, 350), 50, (0, 0, 255), 3)

# POLYGON (triangle example)
pts = np.array([[250, 300], [300, 400], [200, 400]], np.int32)
pts = pts.reshape((-1, 1, 2))

cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=3)

# Filled polygon
cv2.fillPoly(img, [pts], color=(255, 255, 0))

# TEXT
cv2.putText(img, "OpenCV", (150, 480),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# SHOW IMAGE
def safe_imshow(name, image):
    if not args.show:
        return
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image)
    except Exception as e:
        print(f"Warning: can't show window '{name}': {e}")

# Show image (only if --show)
safe_imshow("Drawing Shapes", img)

# SAVE OUTPUT
out_path = OUTPUT_DIR / 'drawing.jpg'
ok = cv2.imwrite(str(out_path), img)
print(f"Saved drawing to: {out_path} -> {ok}")

if args.show:
    try:
        while True:
            try:
                if cv2.getWindowProperty('Drawing Shapes', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
cv2.waitKey(0)
cv2.destroyAllWindows()
