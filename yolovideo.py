import cv2
import cvzone
import math
import time
from ultralytics import YOLO

# Path to the YOLOv8 model weights
model_weights_path = "../Yolo-Weights/yolov8m.pt"

# Path to the input video file
input_video_path = "dataset1.mp4"

# Path to the output video file
output_video_path = "output_video.mp4"

# Load YOLOv8 model
model = YOLO(model_weights_path)

# Define class names
classNames = model.names

prev_frame_time = 0
new_frame_time = 0

# Open the input video file
cap = cv2.VideoCapture(input_video_path)

# Get video frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object to write processed frames into a new video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to use for the output video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    new_frame_time = time.time()
    success, img = cap.read()
    if not success:
        break

    # Perform object detection with YOLOv8
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes  # Access bounding boxes using the 'boxes' attribute
        for box in boxes:
            box_data = box.xyxy[0].tolist()
            print(box_data)  # Print the box data to see its structure

            # Unpack the box data
            x1, y1, x2, y2 = box_data
            conf = box.conf.item()
            cls = int(box.cls.item())

            # Convert class index to class name
            class_name = classNames[cls]
            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
            # Display class name and confidence
            cvzone.putTextRect(img, f'{class_name} {conf:.2f}', (int(x1), int(y1 - 10)), scale=1, thickness=1)

    # Calculate FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS
    cv2.putText(img, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the processed frame to the output video file
    out.write(img)

    # Display the processed frame
    cv2.imshow("Processed Video", img)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and video writer objects
cap.release()
out.release()

# Open the output video file using cv2.VideoCapture to play the processed video
out_cap = cv2.VideoCapture(output_video_path)

while out_cap.isOpened():
    success, frame = out_cap.read()
    if not success:
        break

    cv2.imshow("Processed Video", frame)

    # Check for 'q' key to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
out_cap.release()
cv2.destroyAllWindows()