from collections.abc import Iterable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from motion_estimation.types import BGR2gray, BGRImage, GrayImage, gray2BGR

type Mask = NDArray[np.bool_]


# 1. Define the protocol
class Processor(Protocol):
    def initialize(self, frame: GrayImage) -> None: ...
    def apply(self, frame: GrayImage) -> GrayImage: ...


def process_video_helper(
    video: Iterable[BGRImage],
    processor: Processor,
) -> Iterable[BGRImage]:
    for i, frame in enumerate(video):
        frame = BGR2gray(frame)
        if i == 0:
            processor.initialize(frame)
            continue

        mask = processor.apply(frame)
        yield gray2BGR(mask)
