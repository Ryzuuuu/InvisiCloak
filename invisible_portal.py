# ============================================================
# INVISIBLE PORTAL USING HAND GESTURES
# ============================================================
# This program uses your webcam to create an "invisibility portal".
# Make a rectangle with your thumbs and index fingers (both hands),
# and everything inside that rectangle becomes invisible — replaced
# by the background captured when the program started.
# ============================================================


# ============================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================

import cv2                # OpenCV — for webcam capture, drawing, image processing
import mediapipe as mp    # MediaPipe — for hand detection and finger tracking
import numpy as np        # NumPy — for array operations on images


# ============================================================
# SECTION 2: INITIALIZE MEDIAPIPE HANDS
# ============================================================

# mp.solutions.hands gives us the hand detection model
mp_hands = mp.solutions.hands

# Create a Hands detector object with these settings:
#   - static_image_mode=False : we are processing a video stream (not single images)
#   - max_num_hands=2         : detect up to 2 hands (we need both)
#   - min_detection_confidence=0.7 : how confident the model must be to detect a hand
#   - min_tracking_confidence=0.5  : how confident the model must be to keep tracking
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# mp.solutions.drawing_utils helps draw hand landmarks on the frame
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# SECTION 3: INITIALIZE WEBCAM
# ============================================================

# Open the default webcam (index 0)
webcam = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not webcam.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

# Read one frame to get the dimensions (height and width)
success, test_frame = webcam.read()
if not success:
    print("ERROR: Could not read from webcam.")
    webcam.release()
    exit()

# Get frame dimensions — we'll need these later for coordinate conversion
frame_height, frame_width, _ = test_frame.shape
print(f"Webcam resolution: {frame_width} x {frame_height}")


# ============================================================
# SECTION 4: CAPTURE BACKGROUND
# ============================================================
# We capture the background BEFORE the user steps into the frame.
# This background will be shown inside the portal to create
# the "invisibility" effect.

print("==============================================")
print("  CAPTURING BACKGROUND IN 3 SECONDS...")
print("  Please step out of the camera frame!")
print("==============================================")

# Wait 3 seconds so the user can move out of frame
cv2.waitKey(3000)

# Capture multiple frames and use the last one — this helps the
# camera auto-adjust brightness and focus for a better background
background = None
for i in range(30):
    success, background = webcam.read()

# Flip the background horizontally so it matches the mirror view
# (webcam images are mirrored by default — flipping makes it natural)
background = cv2.flip(background, 1)

# Check if background was captured
if background is None:
    print("ERROR: Could not capture background.")
    webcam.release()
    exit()

print("Background captured successfully!")
print("Now step into the frame and use both hands to create a portal.")
print("Press 'q' to quit.")


# ============================================================
# SECTION 5: HELPER FUNCTION — GET FINGERTIP COORDINATES
# ============================================================
# This function takes a hand's landmarks and returns the (x, y)
# pixel positions of the THUMB TIP and INDEX FINGER TIP.

def get_fingertip_positions(hand_landmarks, width, height):
    """
    Extract thumb tip and index finger tip positions from hand landmarks.
    
    Parameters:
        hand_landmarks : the landmarks object from MediaPipe for one hand
        width          : frame width in pixels
        height         : frame height in pixels
    
    Returns:
        thumb_tip  : (x, y) pixel coordinates of the thumb tip
        index_tip  : (x, y) pixel coordinates of the index finger tip
    """
    # MediaPipe landmark IDs:
    #   4  = THUMB_TIP
    #   8  = INDEX_FINGER_TIP
    # Landmarks are normalized (0.0 to 1.0), so we multiply by frame size
    # to get actual pixel coordinates.

    # Get thumb tip position
    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_x = int(thumb.x * width)
    thumb_y = int(thumb.y * height)
    thumb_tip = (thumb_x, thumb_y)

    # Get index finger tip position
    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_x = int(index.x * width)
    index_y = int(index.y * height)
    index_tip = (index_x, index_y)

    return thumb_tip, index_tip


# ============================================================
# SECTION 6: MAIN LOOP — PROCESS EACH FRAME
# ============================================================

while True:
    # ---------------------------------------------------------
    # STEP 6a: Read a frame from the webcam
    # ---------------------------------------------------------
    success, frame = webcam.read()
    if not success:
        print("ERROR: Could not read frame.")
        break

    # Flip the frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)

    # ---------------------------------------------------------
    # STEP 6b: Convert frame to RGB for MediaPipe
    # ---------------------------------------------------------
    # OpenCV uses BGR color format, but MediaPipe needs RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------
    # STEP 6c: Detect hands using MediaPipe
    # ---------------------------------------------------------
    # Process the RGB frame to find hands
    results = hands_detector.process(rgb_frame)

    # ---------------------------------------------------------
    # STEP 6d: Check if BOTH hands are detected
    # ---------------------------------------------------------
    # results.multi_hand_landmarks is a list of detected hands
    # It is None if no hands are found
    both_hands_detected = (
        results.multi_hand_landmarks is not None
        and len(results.multi_hand_landmarks) >= 2
    )

    if both_hands_detected:
        # ---------------------------------------------------------
        # STEP 6e: Get fingertip positions for both hands
        # ---------------------------------------------------------
        # Hand 0 (first hand detected)
        hand_1 = results.multi_hand_landmarks[0]
        thumb_1, index_1 = get_fingertip_positions(hand_1, frame_width, frame_height)

        # Hand 1 (second hand detected)
        hand_2 = results.multi_hand_landmarks[1]
        thumb_2, index_2 = get_fingertip_positions(hand_2, frame_width, frame_height)

        # ---------------------------------------------------------
        # STEP 6f: Create portal rectangle from the four fingertips
        # ---------------------------------------------------------
        # Collect all four fingertip x and y coordinates
        all_x = [thumb_1[0], index_1[0], thumb_2[0], index_2[0]]
        all_y = [thumb_1[1], index_1[1], thumb_2[1], index_2[1]]

        # Find the bounding rectangle:
        #   top-left corner  = (minimum x, minimum y)
        #   bottom-right corner = (maximum x, maximum y)
        portal_top_left_x = min(all_x)
        portal_top_left_y = min(all_y)
        portal_bottom_right_x = max(all_x)
        portal_bottom_right_y = max(all_y)

        # Make sure the portal has a minimum size (avoid tiny rectangles)
        portal_width = portal_bottom_right_x - portal_top_left_x
        portal_height = portal_bottom_right_y - portal_top_left_y

        if portal_width > 20 and portal_height > 20:
            # ---------------------------------------------------------
            # STEP 6g: Create a mask for the portal region
            # ---------------------------------------------------------
            # A mask is a black-and-white image:
            #   - White (255) = where the portal is (will show background)
            #   - Black (0)   = everywhere else (will show normal frame)

            # Start with a completely black mask (same size as frame)
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Draw a filled white rectangle on the mask for the portal area
            cv2.rectangle(
                mask,                                                   # the mask image
                (portal_top_left_x, portal_top_left_y),                 # top-left corner
                (portal_bottom_right_x, portal_bottom_right_y),         # bottom-right corner
                255,                                                    # color = white (255)
                -1                                                      # thickness = -1 means filled
            )

            # ---------------------------------------------------------
            # STEP 6h: Replace portal region with background
            # ---------------------------------------------------------
            # We use the mask to combine two images:
            #   - Where mask is white → show the background
            #   - Where mask is black → show the current frame

            # cv2.bitwise_and keeps pixels where the mask is white
            # Extract the background region inside the portal
            background_region = cv2.bitwise_and(background, background, mask=mask)

            # Invert the mask: white becomes black and black becomes white
            inverted_mask = cv2.bitwise_not(mask)

            # Extract the current frame region OUTSIDE the portal
            frame_region = cv2.bitwise_and(frame, frame, mask=inverted_mask)

            # Combine both regions: background inside portal + frame outside
            frame = cv2.add(background_region, frame_region)

            # ---------------------------------------------------------
            # STEP 6i: Draw green rectangle around the portal
            # ---------------------------------------------------------
            cv2.rectangle(
                frame,                                                  # draw on the output frame
                (portal_top_left_x, portal_top_left_y),                 # top-left corner
                (portal_bottom_right_x, portal_bottom_right_y),         # bottom-right corner
                (255, 0, 0),                                            # color = green (BGR)
                2                                                       # thickness = 2 pixels
            )

            # ---------------------------------------------------------
            # STEP 6j: Display "Portal Active" text
            # ---------------------------------------------------------
            cv2.putText(
                frame,                          # image to draw on
                "Portal Active",                # text to display
                (portal_top_left_x, portal_top_left_y - 10),  # position (above the portal)
                cv2.FONT_HERSHEY_SIMPLEX,       # font style
                0.8,                            # font scale (size)
                (0, 255, 0),                    # color = green (BGR)
                2                               # thickness of the text
            )

        # ---------------------------------------------------------
        # STEP 6k: Draw hand landmarks on the frame (optional, helpful for debugging)
        # ---------------------------------------------------------
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,                                  # image to draw on
                hand_landmarks,                         # the hand's landmarks
                mp_hands.HAND_CONNECTIONS              # connections between landmarks
            )

    # ---------------------------------------------------------
    # STEP 6l: Display the final output frame
    # ---------------------------------------------------------
    cv2.imshow("Invisible Portal - Press Q to Quit", frame)

    # ---------------------------------------------------------
    # STEP 6m: Check for quit key
    # ---------------------------------------------------------
    # cv2.waitKey(1) waits 1 millisecond and returns the key pressed
    # & 0xFF ensures compatibility across different operating systems
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("Exiting program...")
        break


# ============================================================
# SECTION 7: CLEANUP — RELEASE RESOURCES
# ============================================================

# Release the webcam so other programs can use it
webcam.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Close the MediaPipe hands detector to free memory
hands_detector.close()

print("Program ended. Goodbye!")
