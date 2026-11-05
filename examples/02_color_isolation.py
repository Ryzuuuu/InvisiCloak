"""Lesson 2: isolate blue objects in the webcam feed using an HSV colour mask.

This is the same masking idea the invisible portal builds on.
Press 'q' to quit.
"""

import cv2
import numpy as np

# HSV range that covers blue (OpenCV hue runs from 0 to 179).
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([130, 255, 255])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open webcam.")

print("Starting video loop. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # HSV separates colour (hue) from brightness, which makes thresholding easier.
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # White where the pixel is inside the blue range, black everywhere else.
    mask = cv2.inRange(hsv_frame, LOWER_BLUE, UPPER_BLUE)

    # Keep only the original pixels where the mask is white.
    isolated_color_result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("1. Original Webcam Feed", frame)
    cv2.imshow("2. Binary Mask (Isolated)", mask)
    cv2.imshow("3. Extracted Target Color", isolated_color_result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
