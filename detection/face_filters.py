import cv2
import os

# Load cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# Load glasses image with alpha channel
base_path = os.path.dirname(__file__)
img_path = os.path.join(base_path, "frame.png")

glasses = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if glasses is None:
    print("Error: frame.png not found")
    exit()

if glasses.shape[2] != 4:
    print("Error: PNG must have alpha channel")
    exit()

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

cv2.namedWindow("Face Filters", cv2.WINDOW_NORMAL)

# Smoothing variables
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            # Take first two eyes
            eye1, eye2 = eyes[:2]

            # Sort left and right
            if eye1[0] < eye2[0]:
                left_eye, right_eye = eye1, eye2
            else:
                left_eye, right_eye = eye2, eye1

            lx, ly, lw, lh = left_eye
            rx, ry, rw, rh = right_eye

            # Eye centers (global coordinates)
            left_center = (x + lx + lw // 2, y + ly + lh // 2)
            right_center = (x + rx + rw // 2, y + ry + rh // 2)

            # Distance between eyes
            eye_width = right_center[0] - left_center[0]

            # Resize glasses
            glasses_width = int(eye_width * 2)
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])

            glasses_resized = cv2.resize(glasses, (glasses_width, glasses_height))

            # Position glasses
            x_offset = left_center[0] - int(glasses_width * 0.25)
            y_offset = left_center[1] - int(glasses_height / 2)

            # ===== SMOOTHING =====
            x_offset = int(0.7 * prev_x + 0.3 * x_offset)
            y_offset = int(0.7 * prev_y + 0.3 * y_offset)
            prev_x, prev_y = x_offset, y_offset

            h_g, w_g, _ = glasses_resized.shape

            # Boundary check
            if (x_offset < 0 or y_offset < 0 or
                x_offset + w_g > frame.shape[1] or
                y_offset + h_g > frame.shape[0]):
                continue

            roi = frame[y_offset:y_offset + h_g, x_offset:x_offset + w_g]

            # Split channels
            glasses_rgb = glasses_resized[:, :, :3]
            alpha = glasses_resized[:, :, 3] / 255.0
            alpha = cv2.merge([alpha, alpha, alpha])

            # Blend
            blended = (alpha * glasses_rgb + (1 - alpha) * roi).astype('uint8')

            frame[y_offset:y_offset + h_g, x_offset:x_offset + w_g] = blended

    cv2.imshow("Face Filters", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()