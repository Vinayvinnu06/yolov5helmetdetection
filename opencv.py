import cv2
import numpy as np

# Load the pre-trained deep learning model
model = cv2.dnn.readNetFromCaffe("helmet_detection_model.prototxt", "helmet_detection_weights.caffemodel")

# Define the input shape for the model
input_shape = (300, 300, 3)

# Load the video
video = cv2.VideoCapture("video.mp4")

# Loop over the frames of the video
while True:
    # Read the next frame of the video
    ret, frame = video.read()

    # Break if the video has ended
    if not ret:
        break

    # Resize the frame to the input shape of the model
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))

    # Convert the frame to a 4-dimensional blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, input_shape, (104.0, 177.0, 123.0))

    # Pass the blob through the model
    model.setInput(blob)
    output = model.forward()

    # Loop over the detections in the output
    for i in range(0, output.shape[2]):
        confidence = output[0, 0, i, 2]

        # Filter out detections with low confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box
            box = output[0, 0, i, 3:7] * np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Draw the bounding box around the helmet
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


