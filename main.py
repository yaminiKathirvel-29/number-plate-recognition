import cv2
import pytesseract
import pandas as pd
import os
import re
from datetime import datetime
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

# Your trained YOLO model
MODEL_PATH = "yolov8n.pt"

# Camera:
# 0 = laptop built-in camera
# 1 = Iriun Webcam (usually)
CAMERA_INDEX = 0

# Confidence threshold for YOLO
CONFIDENCE = 0.40

# CSV file where detected plates will be stored
CSV_FILE = "detected_plates.csv"


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# ============================================================
# TESSERACT OCR PATH
# ============================================================

# If Tesseract is already in PATH, this line can be commented.
# Common Windows installation location:

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# CREATE CSV FILE IF IT DOES NOT EXIST
# ============================================================

if not os.path.exists(CSV_FILE):

    df = pd.DataFrame(
        columns=[
            "Number_Plate",
            "Date",
            "Time"
        ]
    )

    df.to_csv(CSV_FILE, index=False)


# ============================================================
# OPEN CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():

    print("ERROR: Camera could not be opened.")
    print("Try changing CAMERA_INDEX from 0 to 1.")
    exit()


print("Camera opened successfully!")
print("Show a car image with a visible number plate.")
print("Press Q to quit.")


# ============================================================
# FUNCTION TO CLEAN OCR RESULT
# ============================================================

def clean_plate(text):

    # Convert to uppercase
    text = text.upper()

    # Remove spaces, dots, hyphens and unwanted characters
    text = re.sub(r'[^A-Z0-9]', '', text)

    return text


# ============================================================
# FUNCTION TO SAVE NUMBER PLATE
# ============================================================

def save_plate(plate):

    if len(plate) < 4:
        return

    # Read existing CSV
    df = pd.read_csv(CSV_FILE)

    # Don't repeatedly save the same plate
    if plate in df["Number_Plate"].astype(str).values:

        return

    now = datetime.now()

    new_data = pd.DataFrame([
        {
            "Number_Plate": plate,
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M:%S")
        }
    ])

    new_data.to_csv(
        CSV_FILE,
        mode="a",
        header=False,
        index=False
    )

    print("----------------------------------")
    print("NUMBER PLATE DETECTED:", plate)
    print("SAVED TO:", CSV_FILE)
    print("----------------------------------")


# ============================================================
# MAIN CAMERA LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read camera frame.")
        break


    # --------------------------------------------------------
    # YOLO NUMBER PLATE DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    # --------------------------------------------------------
    # PROCESS YOLO RESULTS
    # --------------------------------------------------------

    for result in results:

        boxes = result.boxes

        if boxes is None:
            continue


        for box in boxes:

            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)


            # Make sure coordinates are inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            # ------------------------------------------------
            # CROP NUMBER PLATE
            # ------------------------------------------------

            plate_crop = frame[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue


            # ------------------------------------------------
            # PREPROCESS IMAGE FOR OCR
            # ------------------------------------------------

            gray = cv2.cvtColor(
                plate_crop,
                cv2.COLOR_BGR2GRAY
            )

            # Increase size
            gray = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            # Reduce noise
            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            # Threshold
            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7"
            )


            # Clean OCR result
            plate_number = clean_plate(text)


            # ------------------------------------------------
            # DRAW YOLO BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DISPLAY OCR RESULT
            # ------------------------------------------------

            if len(plate_number) >= 4:

                cv2.putText(
                    frame,
                    "Plate: " + plate_number,
                    (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Save to CSV
                save_plate(plate_number)


    # --------------------------------------------------------
    # DISPLAY CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "Number Plate Detection + OCR",
        frame
    )


    # --------------------------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break


# ============================================================
# CLOSE EVERYTHING
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Program stopped.")