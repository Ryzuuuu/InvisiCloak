"""Invisible Portal: hide whatever is inside a rectangle you form with both hands.

The webcam background is captured once at start-up. While the program runs,
MediaPipe tracks both hands and the bounding box of the four fingertips (thumb
tip and index fingertip of each hand) becomes the "portal". Pixels inside the
portal are replaced with the saved background, so anything standing in front of
it appears to vanish.

Controls: press 'q' (or Esc) to quit.
"""

import sys
import time

import cv2
import mediapipe as mp

CAMERA_INDEX = 0
WINDOW_TITLE = "Invisible Portal - Press Q to Quit"

COUNTDOWN_SECONDS = 3  # time to step out of the frame before the background is taken
WARMUP_FRAMES = 30  # frames discarded so auto-exposure and focus can settle
MIN_PORTAL_SIZE = 20  # portal must be larger than this (pixels) in both dimensions
PORTAL_COLOR = (0, 255, 0)  # BGR green
QUIT_KEYS = (ord("q"), ord("Q"), 27)  # 27 = Esc

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def read_mirrored_frame(webcam):
    """Read one frame and flip it horizontally so the view behaves like a mirror.

    Returns None if the camera did not deliver a frame.
    """
    success, frame = webcam.read()
    if not success or frame is None:
        return None
    return cv2.flip(frame, 1)


def capture_background(webcam):
    """Give the user time to step out of view, then grab a clean background frame.

    Returns the mirrored background image, or None if the camera stopped
    delivering frames.
    """
    print("==============================================")
    print(f"  CAPTURING BACKGROUND IN {COUNTDOWN_SECONDS} SECONDS...")
    print("  Please step out of the camera frame!")
    print("==============================================")
    time.sleep(COUNTDOWN_SECONDS)

    # Read several frames and keep the last one: the first few are stale or
    # badly exposed while the camera adjusts.
    background = None
    for _ in range(WARMUP_FRAMES):
        background = read_mirrored_frame(webcam)
        if background is None:
            return None
    return background


def get_fingertip_positions(hand_landmarks, width, height):
    """Return the (x, y) pixel positions of the thumb tip and index fingertip.

    MediaPipe landmarks are normalized to 0.0-1.0, so they are scaled by the
    frame size to get pixel coordinates.
    """
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = (int(thumb.x * width), int(thumb.y * height))
    index_tip = (int(index.x * width), int(index.y * height))
    return thumb_tip, index_tip


def compute_portal_rect(points, width, height, min_size=MIN_PORTAL_SIZE):
    """Return the portal as (x1, y1, x2, y2), or None if it is too small.

    The portal is the bounding box of the given fingertip points, clamped to
    the frame so fingertips that leave the picture cannot index out of range.
    """
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x1, x2 = max(min(xs), 0), min(max(xs), width - 1)
    y1, y2 = max(min(ys), 0), min(max(ys), height - 1)

    if x2 - x1 <= min_size or y2 - y1 <= min_size:
        return None
    return x1, y1, x2, y2


def apply_portal(frame, background, rect):
    """Copy the background into the portal region of the frame (in place)."""
    x1, y1, x2, y2 = rect
    frame[y1 : y2 + 1, x1 : x2 + 1] = background[y1 : y2 + 1, x1 : x2 + 1]
    return frame


def draw_portal_overlay(frame, rect):
    """Outline the portal and label it."""
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), PORTAL_COLOR, 2)
    cv2.putText(
        frame,
        "Portal Active",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        PORTAL_COLOR,
        2,
    )


def run_portal_loop(webcam, hands, background):
    """Show the live feed, applying the portal whenever both hands are visible."""
    while True:
        frame = read_mirrored_frame(webcam)
        if frame is None:
            print("ERROR: Could not read frame.")
            break
        height, width = frame.shape[:2]

        # OpenCV frames are BGR but MediaPipe expects RGB.
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detected_hands = results.multi_hand_landmarks or []

        if len(detected_hands) >= 2:
            points = []
            for hand in detected_hands[:2]:
                points.extend(get_fingertip_positions(hand, width, height))

            rect = compute_portal_rect(points, width, height)
            if rect is not None:
                apply_portal(frame, background, rect)
                draw_portal_overlay(frame, rect)

            for hand in detected_hands:
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF in QUIT_KEYS:
            print("Exiting program...")
            break


def main():
    """Open the webcam, capture the background and run the portal until quit."""
    webcam = cv2.VideoCapture(CAMERA_INDEX)
    if not webcam.isOpened():
        print("ERROR: Could not open webcam.")
        return 1

    try:
        background = capture_background(webcam)
        if background is None:
            print("ERROR: Could not capture background.")
            return 1

        height, width = background.shape[:2]
        print(f"Webcam resolution: {width} x {height}")
        print("Background captured successfully!")
        print("Now step into the frame and use both hands to create a portal.")
        print("Press 'q' to quit.")

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        ) as hands:
            run_portal_loop(webcam, hands, background)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        webcam.release()
        cv2.destroyAllWindows()

    print("Program ended. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
