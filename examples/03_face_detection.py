"""Lesson 3: detect faces in the webcam feed with OpenCV's Haar cascade.

Press 'q' to quit.
"""

import cv2

# Classic frontal-face classifier that ships with OpenCV.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # The cascade works on grayscale images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        f"Count: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    cv2.imshow("Lesson 3: Live Face Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
