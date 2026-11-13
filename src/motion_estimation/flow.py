from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import NoneType
from typing import Any, Literal, Unpack, overload

import cv2
import numpy as np

from motion_estimation.types import (
    DISConfig,
    FarnebackConfig,
    Flow,
    FlowAlgorithmConfig,
    GrayImage,
    SparseToDenseConfig,
    parse_NDArray,
)


@overload
def _init_flow(
    previous: GrayImage,
    current: GrayImage,
) -> Flow: ...
@overload
def _init_flow[Shape: tuple[Any, ...]](
    previous: np.ndarray[Shape, np.dtype[np.uint8]],
    current: np.ndarray[Shape, np.dtype[np.uint8]],
) -> np.ndarray[tuple[Unpack[Shape], Literal[2]], np.dtype[np.float32]]: ...  # noqa: UP044
def _init_flow[Shape: tuple[Any, ...]](
    previous: np.ndarray[Shape, np.dtype[np.uint8]],
    current: np.ndarray[Shape, np.dtype[np.uint8]],
) -> np.ndarray[tuple[Unpack[Shape], Literal[2]], np.dtype[np.float32]]:  # noqa: UP044
    if previous.shape != current.shape:
        raise ValueError("Input images must have identical dimensions")

    return np.zeros((*previous.shape, 2), dtype=np.float32)


def cv2_algorithm_wrapper[ConfigT: FlowAlgorithmConfig](
    config_type: type[ConfigT],
) -> Callable[
    [Callable[[GrayImage, GrayImage, Flow, ConfigT], None]],
    Callable[[GrayImage, GrayImage, ConfigT | None], Flow],
]:
    def decorator(
        calc: Callable[[GrayImage, GrayImage, Flow, ConfigT], None],
    ) -> Callable[[GrayImage, GrayImage, ConfigT | None], Flow]:

        @wraps(calc)
        def wrapper(
            previous: GrayImage,
            current: GrayImage,
            config: ConfigT | None = None,
        ) -> Flow:
            if config is None:
                config = config_type()

            flow = _init_flow(previous, current)

            calc(previous, current, flow, config)

            return Flow(parse_NDArray(flow, np.float32))

        return wrapper

    return decorator


@cv2_algorithm_wrapper(FarnebackConfig)
def farneback(
    previous: GrayImage,
    current: GrayImage,
    flow: Flow,
    config: FarnebackConfig,
) -> None:
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


@cv2_algorithm_wrapper(DISConfig)
def dis(
    previous: GrayImage,
    current: GrayImage,
    flow: Flow,
    config: DISConfig,
) -> None:
    cv2.DISOpticalFlow.create(config.preset).calc(
        previous,
        current,
        flow,
    )


@cv2_algorithm_wrapper(NoneType)
def dual_tvl1(
    previous: GrayImage,
    current: GrayImage,
    flow: Flow,
    config: None,
) -> None:
    cv2.optflow.DualTVL1OpticalFlow.create().calc(
        previous,
        current,
        flow,
    )


@cv2_algorithm_wrapper(NoneType)
def deep_flow(
    previous: GrayImage,
    current: GrayImage,
    flow: Flow,
    config: None,
) -> None:
    cv2.optflow.createOptFlow_DeepFlow().calc(
        previous,
        current,
        flow,
    )


@cv2_algorithm_wrapper(NoneType)
def pca_flow(
    previous: GrayImage,
    current: GrayImage,
    flow: Flow,
    config: None,
) -> None:
    cv2.optflow.createOptFlow_PCAFlow().calc(
        previous,
        current,
        flow,
    )


@cv2_algorithm_wrapper(SparseToDenseConfig)
def sparse_to_dense(
    previous: GrayImage,
    current: GrayImage,
    flow: Flow,
    config: SparseToDenseConfig,
) -> None:
    cv2.optflow.calcOpticalFlowSparseToDense(
        previous,
        current,
        flow,
        grid_step=config.grid_step,
        k=config.k,
        sigma=config.sigma,
        use_post_proc=config.use_post_proc,
        fgs_lambda=config.fgs_lambda,
        fgs_sigma=config.fgs_sigma,
    )
