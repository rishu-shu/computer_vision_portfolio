import cv2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

mode = "normal"

from pathlib import Path
from datetime import datetime

print("Controls:")
print("g = grayscale | b = blur | e = edges | n = normal | s = save frame | q = quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    display = frame.copy()

    # ===== MODES =====
    if mode == "gray":
        display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    elif mode == "blur":
        display = cv2.GaussianBlur(display, (15, 15), 0)

    elif mode == "edges":
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        display = cv2.Canny(gray, 100, 200)

    # If display is single-channel (gray or edges), convert to BGR for consistent text/color
    if len(display.shape) == 2 or display.shape[2] == 1:
        display_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    else:
        display_bgr = display

    # ===== TEXT =====
    cv2.putText(display_bgr, f"Mode: {mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(display_bgr, "Press 'g' (gray), 'b' (blur), 'e' (edges), 'n' (normal), 's' (save), 'q' (quit)",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Webcam", display_bgr)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('g'):
        mode = "gray"

    elif key == ord('b'):
        mode = "blur"

    elif key == ord('e'):
        mode = "edges"

    elif key == ord('n'):
        mode = "normal"

    elif key == ord('q'):
        break
    elif key == ord('s'):
        # Save current displayed frame to script-relative outputs folder
        SCRIPT_DIR = Path(__file__).resolve().parent
        OUT_DIR = SCRIPT_DIR / "outputs"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = OUT_DIR / f"webcam_{ts}.png"
        # save the BGR image (display_bgr is always BGR)
        ok = cv2.imwrite(str(out_path), display_bgr)
        print(f"Saved frame to: {out_path} -> {ok}")

# Release resources
cap.release()
cv2.destroyAllWindows()
