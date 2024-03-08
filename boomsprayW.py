import cv2
import numpy as np
from ultralytics import YOLO

# Load trained model
model = YOLO(model=r"C:\Users\K BHANU PRASAD\Desktop\orchad-crop\weights\weights\best.pt", task='detect')

# Video capture
cap = cv2.VideoCapture(r"C:\Users\K BHANU PRASAD\Desktop\orchad-crop\VID_20240301_144236.mp4")

# Define the new frame width and height
new_width = 640
new_height = 480

# Calculate the coordinates for the line
line_color = (0, 255, 0)  # Green color
line_thickness = 2
line_position = new_width // 2

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('boomspray_output.avi', fourcc, 25.0, (new_width, new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frameq
    frame = cv2.resize(frame, (new_width, new_height))

    detections1 = model(frame)
    # Draw a line in the middle of the frame
    frame = cv2.line(frame, (line_position, 0), (line_position, new_height), line_color, line_thickness)
    boomspray_on = False
    for detections in detections1:
        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < 0.5:
                continue
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            if (xmin + xmax) // 2 > line_position:
                boomspray_on = True

    # Print "Boomspray is on" or "Boomspray is off" in the frame
    if boomspray_on:
        cv2.putText(frame, "Boomspray is on", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Boomspray is off", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display result
    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
