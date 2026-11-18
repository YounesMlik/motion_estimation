from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from motion_estimation.flow import GrayImage
from motion_estimation.types import BGRImage, RGBImage, parse_NDArray


@contextmanager
def open_video(
    path: str | Path,
) -> Generator[cv2.VideoCapture]:
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
) -> Iterable[BGRImage]:
    """Yield resized BGR frames from a video."""

    with open_video(path) as capture:
        while True:
            success, frame = capture.read()

            if not success:
                break

            yield BGRImage(parse_NDArray(frame, np.uint8))


def read_video_from_jpeg_frames(
    folder_path: str | Path,
) -> Iterable[BGRImage]:
    """Yield BGR frames from JPEG images in a folder."""
    paths = sorted(
        path
        for path in Path(folder_path).iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg"}
    )

    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")

        yield BGRImage(parse_NDArray(frame, np.uint8))


def show_video(frames: Iterable[BGRImage], *, fps: int = 30, title="Video"):
    delay = round(1000 / fps)
    try:
        for frame in frames:
            cv2.imshow(title, frame)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    except cv2.error:
        pass
    finally:
        cv2.destroyAllWindows()
