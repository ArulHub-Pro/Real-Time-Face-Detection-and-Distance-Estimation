import cv2

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Known parameters for calibration
KNOWN_WIDTH = 14.0  # Known width of the face in cm (example)
KNOWN_DISTANCE = 50.0  # Known distance from the camera in cm

# Function to calculate focal length based on known distance and object width
def calculate_focal_length(known_distance, known_width, width_in_pixels):
    return (width_in_pixels * known_distance) / known_width

# Placeholder for the focal length (calculated during calibration)
FOCAL_LENGTH = None

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'c' to calibrate focal length using a reference face.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale and apply preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display pixel dimensions directly on the bounding box
        pixel_dimensions = f"Width: {w}px, Height: {h}px"
        cv2.putText(frame, pixel_dimensions, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if FOCAL_LENGTH:
            # Calculate real-world size and distance
            real_width = (KNOWN_WIDTH * FOCAL_LENGTH) / w  # Width in cm
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / w    # Distance in cm
            cv2.putText(frame, f"Size: {real_width:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Calibrate using 'c'", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Face Size and Distance Detection", frame)

    # Check for user input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('c'):  # Calibrate focal length
        if len(faces) > 0:
            # Use the first detected face for calibration
            _, _, w, _ = faces[0]
            FOCAL_LENGTH = calculate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, w)
            print(f"Calibration complete! Focal Length: {FOCAL_LENGTH:.2f}")

# Release resources
cap.release()
cv2.destroyAllWindows()
