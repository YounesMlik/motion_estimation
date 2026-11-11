
from dataclasses import dataclass
from typing import NewType

import cv2
import numpy as np
from numpy.typing import NDArray

GrayImage = NewType("GrayImage" ,NDArray[np.uint8])
Flow = NewType("Flow" ,NDArray[np.float32])


@dataclass(frozen=True, slots=True)
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


def farneback(
    previous: GrayImage,
    current: GrayImage,
    config: FarnebackConfig | None = None,
) -> Flow:
    """Estimate dense optical flow between two grayscale images."""
    if config is None:
        config = FarnebackConfig()

    if previous.shape != current.shape:
        raise ValueError("Input images must have identical dimensions")

    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )
