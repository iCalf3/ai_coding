# This script captures an image from the default camera and saves it as 'captured_image.jpg'.
# It also displays the captured image in a window.
# 
# History:
# 2025-05-03: Created by iCalf3

import cv2

def capture_image():
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Display the captured frame
        cv2.imshow("Captured Image", frame)

        # Save the captured image to a file
        cv2.imwrite("captured_image.jpg", frame)
        print("Image saved as 'captured_image.jpg'.")

        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not read frame.")

    # Release the camera
    cap.release()

if __name__ == "__main__":
    capture_image()
