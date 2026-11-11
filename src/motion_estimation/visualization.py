from typing import NewType

import cv2
import numpy as np
from numpy.typing import NDArray

from motion_estimation.types import parse_NDArray

from .flow import Flow

BGRImage = NewType("BGRImage" ,NDArray[np.uint8])


def colorize(
    flow: Flow,
    max_motion: float = 20.0,
) -> BGRImage:
    """Visualize flow direction as hue and magnitude as brightness."""

    if max_motion <= 0:
        raise ValueError("max_motion must be positive")

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
    )

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)

    hsv[..., 0] = (angle * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        magnitude * 255 / max_motion,
        0,
        255,
    ).astype(np.uint8)

    return BGRImage(parse_NDArray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.uint8))
