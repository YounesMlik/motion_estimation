import cv2
import numpy as np

from motion_estimation.types import (
    DISConfig,
    FarnebackConfig,
    Flow,
    GrayImage,
    parse_NDArray,
)


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

    flow = np.zeros(shape=(*previous.shape, 2), dtype=np.float32)
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
    )

    return Flow(parse_NDArray(flow, np.float32))


def dis(
    previous: GrayImage,
    current: GrayImage,
    config: DISConfig | None = None,
) -> Flow:
    """Estimate dense optical flow using DIS."""
    if config is None:
        config = DISConfig()

    if previous.shape != current.shape:
        raise ValueError("Input images must have identical dimensions")

    flow = np.zeros(shape=(*previous.shape, 2), dtype=np.float32)

    cv2.DISOpticalFlow.create(config.preset).calc(
        previous,
        current,
        flow,
    )

    return Flow(parse_NDArray(flow, np.float32))
