from collections.abc import Iterable

import numpy as np
from scipy import ndimage

from motion_detection.utils import process_video_helper
from motion_estimation.types import BGRImage, GrayImage


class FrameDifference:
    def __init__(self, threshold: int = 20):
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be in [0, 255]")

        self.threshold = threshold
        self.previous_frame: GrayImage | None = None

    def reset(self) -> None:
        self.previous_frame = None

    def initialize(self, frame: GrayImage) -> None:
        self.previous_frame = frame

    def apply(self, frame: GrayImage) -> GrayImage:
        if self.previous_frame is None:
            raise RuntimeError("FrameDifference is not initialized")
        if frame.shape != self.previous_frame.shape:
            raise ValueError(
                f"Frame shape changed: {self.previous_frame.shape} -> {frame.shape}"
            )

        difference = np.abs(
            frame.astype(np.int16) - self.previous_frame.astype(np.int16)
        )

        self.previous_frame = frame

        foreground = difference > self.threshold
        foreground = foreground.astype(np.uint8) * 255
        foreground = ndimage.median_filter(foreground, 3)

        return foreground

    @staticmethod
    def process_video(
        video: Iterable[BGRImage],
        threshold: int = 20,
    ) -> Iterable[BGRImage]:
        processor = FrameDifference(threshold)

        return process_video_helper(video, processor)
