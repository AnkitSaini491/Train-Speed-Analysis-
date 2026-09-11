import cv2
import time
from ultralytics import YOLO

# -----------------------------
# SETTINGS
# -----------------------------

VIDEO_PATH = "train.mp4"

# Maximum allowed speed
SPEED_LIMIT = 80

# Camera calibration
# Change this according to your video
PIXELS_PER_METER = 20

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Video not found!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# Previous center position
previous_x = None
previous_y = None
previous_time = None

speed_kmh = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()

    results = model(frame, verbose=False)

    train_found = False

    for result in results:

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # COCO class 6 = train
            # YOLO COCO actually uses class 6 for train
            if class_id != 6:
                continue

            if confidence < 0.50:
                continue

            train_found = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # -----------------------------
            # SPEED CALCULATION
            # -----------------------------

            if previous_x is not None:

                distance_pixels = (
                    (center_x - previous_x) ** 2 +
                    (center_y - previous_y) ** 2
                ) ** 0.5

                distance_meters = (
                    distance_pixels / PIXELS_PER_METER
                )

                time_seconds = current_time - previous_time

                if time_seconds > 0:

                    speed_ms = distance_meters / time_seconds

                    speed_kmh = speed_ms * 3.6

            previous_x = center_x
            previous_y = center_y
            previous_time = current_time

            # -----------------------------
            # DRAW DETECTION
            # -----------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"TRAIN {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # -----------------------------
    # DASHBOARD
    # -----------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (350, 150),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "TRAIN SPEED ANALYSIS",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Speed: {speed_kmh:.1f} km/h",
        (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Limit: {SPEED_LIMIT} km/h",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # OVERSPEED ALERT
    # -----------------------------

    if speed_kmh > SPEED_LIMIT:

        cv2.putText(
            frame,
            "OVERSPEED ALERT!",
            (25, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "STATUS: NORMAL",
            (25, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Train Speed Analysis",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
