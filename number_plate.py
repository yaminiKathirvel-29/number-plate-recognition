from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")

# Read the car image
image = cv2.imread("car.jpg")

if image is None:
    print("IMAGE NOT FOUND!")
    exit()

print("IMAGE FOUND!")

# Detect objects
results = model(image)

# Display the result
for result in results:
    result.show()

print("Detection completed!")