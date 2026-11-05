"""Unit tests for the portal geometry and compositing logic.

They run without a webcam or display: the camera, window and hand detector are
replaced with small fakes.

Run from the repository root:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import invisible_portal as portal  # noqa: E402

WIDTH, HEIGHT = 640, 480


def make_hand(thumb_xy, index_xy):
    """Build a fake MediaPipe hand with the two fingertips we care about."""
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    landmarks[portal.mp_hands.HandLandmark.THUMB_TIP] = SimpleNamespace(
        x=thumb_xy[0], y=thumb_xy[1]
    )
    landmarks[portal.mp_hands.HandLandmark.INDEX_FINGER_TIP] = SimpleNamespace(
        x=index_xy[0], y=index_xy[1]
    )
    return SimpleNamespace(landmark=landmarks)


class FakeWebcam:
    """Returns the same frame forever, or nothing once `frames` runs out."""

    def __init__(self, frame, frames=None):
        self.frame = frame
        self.remaining = frames

    def read(self):
        if self.remaining is not None:
            if self.remaining == 0:
                return False, None
            self.remaining -= 1
        return True, self.frame.copy()


class QuietTestCase(unittest.TestCase):
    """Base class that silences the program's console output during tests."""

    def setUp(self):
        patcher = mock.patch("builtins.print")
        patcher.start()
        self.addCleanup(patcher.stop)


class GetFingertipPositionsTests(QuietTestCase):
    def test_scales_normalized_landmarks_to_pixels(self):
        hand = make_hand((0.5, 0.25), (1.0, 0.0))
        thumb, index = portal.get_fingertip_positions(hand, WIDTH, HEIGHT)
        self.assertEqual(thumb, (320, 120))
        self.assertEqual(index, (640, 0))


class ComputePortalRectTests(unittest.TestCase):
    def test_returns_bounding_box_of_all_points(self):
        points = [(100, 200), (300, 150), (120, 400), (280, 380)]
        self.assertEqual(portal.compute_portal_rect(points, WIDTH, HEIGHT), (100, 150, 300, 400))

    def test_clamps_points_outside_the_frame(self):
        points = [(-50, -20), (700, 100), (10, 600), (300, 300)]
        self.assertEqual(
            portal.compute_portal_rect(points, WIDTH, HEIGHT),
            (0, 0, WIDTH - 1, HEIGHT - 1),
        )

    def test_rejects_portals_that_are_too_small(self):
        narrow = [(100, 100), (120, 300), (110, 200), (115, 250)]  # width == 20
        flat = [(100, 100), (300, 120), (200, 110), (250, 115)]  # height == 20
        self.assertIsNone(portal.compute_portal_rect(narrow, WIDTH, HEIGHT))
        self.assertIsNone(portal.compute_portal_rect(flat, WIDTH, HEIGHT))

    def test_accepts_portal_just_above_minimum_size(self):
        points = [(100, 100), (121, 121)]
        self.assertEqual(portal.compute_portal_rect(points, WIDTH, HEIGHT), (100, 100, 121, 121))


class ApplyPortalTests(unittest.TestCase):
    def test_copies_background_inside_rect_only(self):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        background = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)

        portal.apply_portal(frame, background, (100, 50, 300, 250))

        self.assertTrue((frame[50:251, 100:301] == 200).all())  # edges included
        self.assertEqual(int(frame.sum()), 200 * 3 * 201 * 201)  # nothing else touched


class CaptureBackgroundTests(QuietTestCase):
    def test_returns_mirrored_last_frame(self):
        source = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        source[:, 0] = 255  # bright left-most column
        with mock.patch.object(portal.time, "sleep") as sleep:
            background = portal.capture_background(FakeWebcam(source))

        sleep.assert_called_once_with(portal.COUNTDOWN_SECONDS)
        self.assertTrue((background[:, -1] == 255).all())  # now on the right
        self.assertTrue((background[:, 0] == 0).all())

    def test_returns_none_when_camera_stops_delivering_frames(self):
        source = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        webcam = FakeWebcam(source, frames=5)
        with mock.patch.object(portal.time, "sleep"):
            self.assertIsNone(portal.capture_background(webcam))


class RunPortalLoopTests(QuietTestCase):
    def run_loop(self, hands):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        background = np.full((HEIGHT, WIDTH, 3), 200, dtype=np.uint8)
        shown = []

        with (
            mock.patch.object(portal.cv2, "imshow", lambda _, img: shown.append(img.copy())),
            mock.patch.object(portal.cv2, "waitKey", return_value=ord("q")),
            mock.patch.object(portal.mp_drawing, "draw_landmarks"),
        ):
            portal.run_portal_loop(FakeWebcam(frame), hands, background)
        return shown

    def test_reveals_background_when_both_hands_form_a_portal(self):
        left = make_hand((0.25, 0.25), (0.30, 0.30))
        right = make_hand((0.70, 0.70), (0.75, 0.75))
        hands = SimpleNamespace(
            process=lambda _: SimpleNamespace(multi_hand_landmarks=[left, right])
        )

        shown = self.run_loop(hands)

        self.assertEqual(len(shown), 1)
        self.assertTrue((shown[0][240, 320] == 200).all())  # inside the portal
        self.assertTrue((shown[0][10, 10] == 0).all())  # outside the portal

    def test_leaves_frame_untouched_with_a_single_hand(self):
        hand = make_hand((0.25, 0.25), (0.75, 0.75))
        hands = SimpleNamespace(process=lambda _: SimpleNamespace(multi_hand_landmarks=[hand]))

        shown = self.run_loop(hands)

        self.assertEqual(int(shown[0].sum()), 0)

    def test_leaves_frame_untouched_without_hands(self):
        hands = SimpleNamespace(process=lambda _: SimpleNamespace(multi_hand_landmarks=None))

        shown = self.run_loop(hands)

        self.assertEqual(int(shown[0].sum()), 0)


class MainTests(QuietTestCase):
    def test_fails_cleanly_when_webcam_cannot_be_opened(self):
        webcam = mock.Mock()
        webcam.isOpened.return_value = False
        with mock.patch.object(portal.cv2, "VideoCapture", return_value=webcam):
            self.assertEqual(portal.main(), 1)

    def test_releases_webcam_when_background_capture_fails(self):
        webcam = mock.Mock()
        webcam.isOpened.return_value = True
        webcam.read.return_value = (False, None)
        with (
            mock.patch.object(portal.cv2, "VideoCapture", return_value=webcam),
            mock.patch.object(portal.cv2, "destroyAllWindows"),
            mock.patch.object(portal.time, "sleep"),
        ):
            self.assertEqual(portal.main(), 1)
        webcam.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
