# Invisible Portal

This project uses your webcam and hand gestures to create an "invisible portal" effect with the script [invisible_portal.py](invisible_portal.py).

## What it does

When you make a rectangle with your thumbs and index fingers using both hands, the area inside that rectangle is replaced with the background captured at the start of the program. This creates the illusion that the person inside the portal has become invisible.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If you want to install them manually, the script uses:

- OpenCV
- MediaPipe
- NumPy

## How to run

```bash
python invisible_portal.py
```

## Usage

1. Run the script.
2. The program will ask you to step out of the camera frame so it can capture the background.
3. After the background is captured, step back into the frame.
4. Use both hands to form a rectangle with your thumbs and index fingers.
5. The area inside the rectangle will show the saved background.
6. Press `q` to quit.

## Notes

- A working webcam is required.
- The script works best in a well-lit environment.
- The portal is only created when both hands are detected clearly.
