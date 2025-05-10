import cv2
import os
import time

# Create a directory to save detected images
output_dir = "detected_persons"
os.makedirs(output_dir, exist_ok=True)

# Load pre-trained Haar Cascade for person detection
cascade_path = cv2.data.haarcascades + "haarcascade_upperbody.xml"
person_cascade = cv2.CascadeClassifier(cascade_path)

# Initialize the camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to grayscale for detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect persons in the frame
    persons = person_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected persons and save images
    for (x, y, w, h) in persons:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(output_dir, f"person_{timestamp}.jpg")
        cv2.imwrite(image_path, frame[y:y+h, x:x+w])

    # Display the resulting frame
    cv2.imshow("Person Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()