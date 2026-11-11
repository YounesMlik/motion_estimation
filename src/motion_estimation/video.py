from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import NewType

import cv2
import numpy as np
from numpy.typing import NDArray

from motion_estimation.flow import GrayImage
from motion_estimation.types import parse_NDArray

Image = NewType("Image", NDArray[np.uint8])


@contextmanager
def open_video(
    path: str | Path,
) -> Generator[cv2.VideoCapture, None, None]:
    """Open a video and guarantee that its capture is released."""

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
    scale: float = 1.0,
) -> Generator[Image]:
    """Yield resized BGR frames from a video."""

    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")

    with open_video(path) as capture:
        while True:
            success, frame = capture.read()

            if not success:
                break

            height, width = frame.shape[:2]

            if scale != 1.0:
                size = (
                    round(width * scale),
                    round(height * scale),
                )
                frame = cv2.resize(frame, size)

            yield Image(parse_NDArray(frame, np.uint8))


def to_grayscale(image: Image) -> GrayImage:
    """Convert a BGR image to grayscale."""
    return GrayImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.uint8))
