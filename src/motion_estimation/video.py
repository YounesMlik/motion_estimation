from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from motion_estimation.flow import GrayImage
from motion_estimation.types import BGRImage, RGBImage, parse_NDArray


@contextmanager
def open_video(
    path: str | Path,
) -> Generator[cv2.VideoCapture, None, None]:
    """Open a video and guarantee that its capture is released."""
    if not Path(path).exists():
        raise RuntimeError(f"File does not exist: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        yield capture
    finally:
        capture.release()


def read_video(
    path: str | Path,
) -> Generator[BGRImage]:
    """Yield resized BGR frames from a video."""

    with open_video(path) as capture:
        while True:
            success, frame = capture.read()

            if not success:
                break

            yield BGRImage(parse_NDArray(frame, np.uint8))


def RGB_to_grayscale(image: RGBImage) -> GrayImage:
    """Convert a RGB image to grayscale."""
    return GrayImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), np.uint8))


def BGR_to_grayscale(image: BGRImage) -> GrayImage:
    """Convert a BGR image to grayscale."""
    return GrayImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.uint8))
