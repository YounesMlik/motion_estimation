from typing import Any, Literal, overload

import cv2
import numpy as np

from motion_estimation.types import (
    DISConfig,
    FarnebackConfig,
    Flow,
    GrayImage,
    SparseToDenseConfig,
    parse_NDArray,
)


@overload
def _init_flow(
    previous: GrayImage,
    current: GrayImage,
) -> np.typing.NDArray[np.float32]: ...
@overload
def _init_flow[Shape: tuple[Any, ...]](
    previous: np.ndarray[Shape, np.dtype[np.uint8]],
    current: np.ndarray[Shape, np.dtype[np.uint8]],
) -> np.ndarray[tuple[*Shape, Literal[2]], np.dtype[np.float32]]: ...
def _init_flow[Shape: tuple[Any, ...]](
    previous: np.ndarray[Shape, np.dtype[np.uint8]],
    current: np.ndarray[Shape, np.dtype[np.uint8]],
) -> np.ndarray[tuple[*Shape, Literal[2]], np.dtype[np.float32]]:
    if previous.shape != current.shape:
        raise ValueError("Input images must have identical dimensions")

    return np.zeros((*previous.shape, 2), dtype=np.float32)


def farneback(
    previous: GrayImage,
    current: GrayImage,
    config: FarnebackConfig | None = None,
) -> Flow:
    """Estimate dense optical flow between two grayscale images."""
    if config is None:
        config = FarnebackConfig()

    flow = _init_flow(previous, current)

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

    flow = _init_flow(previous, current)

    cv2.DISOpticalFlow.create(config.preset).calc(
        previous,
        current,
        flow,
    )

    return Flow(parse_NDArray(flow, np.float32))


def dual_tvl1(
    previous: GrayImage,
    current: GrayImage,
    config: None = None,
) -> Flow:
    """Estimate dense optical flow using Dual TV-L1."""

    flow = _init_flow(previous, current)

    cv2.optflow.DualTVL1OpticalFlow.create().calc(
        previous,
        current,
        flow,
    )

    return Flow(parse_NDArray(flow, np.float32))


def deep_flow(
    previous: GrayImage,
    current: GrayImage,
    config: None = None,
) -> Flow:
    """Estimate dense optical flow using DeepFlow."""

    flow = _init_flow(previous, current)

    cv2.optflow.createOptFlow_DeepFlow().calc(
        previous,
        current,
        flow,
    )

    return Flow(parse_NDArray(flow, np.float32))


def pca_flow(
    previous: GrayImage,
    current: GrayImage,
    config: None = None,
) -> Flow:
    """Estimate dense optical flow using PCAFlow."""

    flow = _init_flow(previous, current)

    cv2.optflow.createOptFlow_PCAFlow().calc(
        previous,
        current,
        flow,
    )

    return Flow(parse_NDArray(flow, np.float32))


def sparse_to_dense(
    previous: GrayImage,
    current: GrayImage,
    config: SparseToDenseConfig | None = None,
) -> Flow:
    """Estimate dense optical flow using sparse-to-dense interpolation."""
    if config is None:
        config = SparseToDenseConfig()

    if previous.shape != current.shape:
        raise ValueError("Input images must have identical dimensions")

    flow = cv2.optflow.calcOpticalFlowSparseToDense(
        previous,
        current,
        grid_step=config.grid_step,
        k=config.k,
        sigma=config.sigma,
        use_post_proc=config.use_post_proc,
        fgs_lambda=config.fgs_lambda,
        fgs_sigma=config.fgs_sigma,
    )

    return Flow(parse_NDArray(flow, np.float32))
