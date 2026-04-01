import cv2
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Start webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture")
        break

    # Flip for mirror view (optional)
    frame = cv2.flip(frame, 1)

    # Resize to match training
    img = cv2.resize(frame, (64, 64))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flatten
    flat = gray.flatten().reshape(1, -1)

    # Predict
    prediction = model.predict(flat)[0]

    # Show prediction
    cv2.putText(frame, f"Prediction: {prediction}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Show webcam
    cv2.imshow("Sign Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()