"""Lesson 1: draw shapes and text on a blank NumPy canvas with OpenCV."""

import cv2
import numpy as np

print("OpenCV Version:", cv2.__version__)

# An image is just a NumPy array: 400 rows x 600 columns x 3 colour channels (BGR).
canvas = np.zeros((400, 600, 3), dtype="uint8")

# Paint a red square in the top-left corner by slicing the array directly.
canvas[10:50, 10:50] = [0, 0, 255]

# OpenCV drawing helpers: a blue rectangle outline and a filled green circle.
cv2.rectangle(canvas, (100, 100), (300, 250), (255, 0, 0), 3)
cv2.circle(canvas, (450, 200), 50, (0, 255, 0), -1)

cv2.putText(
    canvas, "Hello Vision!", (100, 320),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
)

cv2.imshow("Lesson 1: Canvas & Matrix", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
