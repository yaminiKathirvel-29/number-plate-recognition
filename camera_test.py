from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working ❌")
        break

    # YOLO detects objects
    results = model(frame)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("YOLO Camera", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()