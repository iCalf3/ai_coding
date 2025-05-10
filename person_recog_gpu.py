import cv2
import pyopenpose as op
import argparse

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_index", type=int, default=0, help="Index of the camera to use.")
    args = parser.parse_args()

    # Set OpenPose parameters
    params = dict()
    params["model_folder"] = "models/"
    params["net_resolution"] = "-1x368"  # Adjust for GPU performance
    params["model_pose"] = "BODY_25"
    params["disable_multi_thread"] = False
    params["num_gpu"] = 1
    params["num_gpu_start"] = 0

    # Initialize OpenPose
    try:
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
    except Exception as e:
        print(f"Error initializing OpenPose: {e}")
        return

    # Open the camera
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Process frame with OpenPose
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop([datum])

        # Display results
        output_frame = datum.cvOutputData
        cv2.imshow("OpenPose Result", output_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()