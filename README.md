# InvisiCloak: Invisible Portal

Form a rectangle with your thumbs and index fingers, and everything inside it disappears.
InvisiCloak uses your webcam and real-time hand tracking to swap the area between your hands
for the empty background it saw a few seconds earlier.

![Python](https://img.shields.io/badge/python-3.9--3.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.21-orange)

## Demo

<!-- Add a short screen recording at docs/demo.gif and uncomment the line below. -->
<!-- ![Invisible portal demo](docs/demo.gif) -->

_Demo GIF coming soon._

## How it works

1. **Capture the background.** On start-up the program counts down three seconds so you can step
   out of view, discards 30 warm-up frames while the camera adjusts its exposure, and keeps the
   last frame as a clean background plate.
2. **Track both hands.** Every frame is converted from BGR to RGB and passed to
   [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker),
   which returns 21 landmarks per hand.
3. **Build the portal.** The thumb tip and index fingertip of each hand give four points. Their
   bounding box, clamped to the frame, is the portal. Boxes 20 pixels or smaller in either
   direction are ignored.
4. **Swap the pixels.** Inside the box the live frame is replaced with the stored background, so
   anything in front of it, including you, is no longer visible. A green outline and a
   "Portal Active" label are drawn on top.

The feed is mirrored horizontally, so moving your hand right moves it right on screen.

## Tech stack

| Purpose                              | Library                                           |
| ------------------------------------ | ------------------------------------------------- |
| Webcam capture, drawing, compositing | [OpenCV](https://opencv.org/)                     |
| Hand detection and landmark tracking | [MediaPipe](https://ai.google.dev/edge/mediapipe) |
| Image arrays                         | [NumPy](https://numpy.org/)                       |

## Getting started

### Prerequisites

- Python 3.9 to 3.12 (MediaPipe 0.10.21 does not publish wheels for newer versions)
- A webcam and a desktop environment that can open a window

### Installation

```bash
git clone https://github.com/Ryzuuuu/InvisiCloak.git
cd InvisiCloak

python -m venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
# macOS / Linux:        source venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python invisible_portal.py
```

## Usage

1. Start the script and **step out of the camera frame**. The background is captured after a
   three-second countdown.
2. Step back in once the console says `Background captured successfully!`.
3. Hold both hands up and make an "L" shape with each thumb and index finger, so the four
   fingertips outline a rectangle.
4. The area inside the rectangle now shows the saved background. Move your hands to resize and
   reposition the portal.
5. Press `q` (or `Esc`) with the video window focused to quit.

### Tips

- Keep the camera still after the background is captured, since a moved camera breaks the illusion.
- Good, even lighting improves hand tracking and keeps the background matching the live feed.
- Both hands must be clearly visible. With only one hand detected the frame is shown unchanged.
- If the wrong camera opens, change `CAMERA_INDEX` at the top of `invisible_portal.py`.

## Project structure

```text
.
├── invisible_portal.py        # the application
├── requirements.txt           # pinned runtime dependencies
├── tests/
│   └── test_invisible_portal.py   # unit tests (no webcam needed)
└── examples/                  # OpenCV practice scripts that lead up to the portal
    ├── 01_canvas_basics.py    # drawing on a NumPy canvas
    ├── 02_color_isolation.py  # HSV colour masking on the webcam feed
    └── 03_face_detection.py   # Haar-cascade face detection
```

## Testing

The tests replace the webcam, window and hand detector with fakes, so they run headless:

```bash
python -m unittest discover -s tests -v
```

## Limitations

- The effect assumes a static scene and a fixed camera. Changes in lighting or a moving background
  show up as a visible seam at the edge of the portal.
- The portal is always an axis-aligned rectangle, even if your hands are tilted.
- Only the first two detected hands are used.
