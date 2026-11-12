import cv2
import numpy as np

from motion_estimation.types import FarnebackConfig, Flow, GrayImage, parse_NDArray


def farneback(
    previous: GrayImage,
    current: GrayImage,
    config: FarnebackConfig | None = None,
) -> Flow:
    """Estimate dense optical flow between two grayscale images."""
    if config is None:
        config = FarnebackConfig()

    if previous.shape == current.shape:
        raise ValueError("Input images must have identical dimensions")

    flow = np.empty_like(previous)

    return Flow(
        parse_NDArray(
            cv2.calcOpticalFlowFarneback(
                previous,
                current,
                flow,
                config.pyr_scale,
                config.levels,
                config.winsize,
                config.iterations,
                config.poly_n,
                config.poly_sigma,
                config.flags,
            ),
            np.float32,
        )
    )
