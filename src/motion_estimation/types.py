from collections.abc import Callable
from dataclasses import dataclass
from types import NoneType
from typing import Any, NewType, overload

import cv2
import numpy as np
from numpy.typing import NDArray

BGRImage = NewType("BGRImage", NDArray[np.uint8])
RGBImage = NewType("RGBImage", NDArray[np.uint8])
GrayImage = NewType("GrayImage", NDArray[np.uint8])
Flow = NewType("Flow", NDArray[np.float32])


@dataclass(slots=True)
class FlowResult:
    frame: BGRImage
    flow: Flow
    visualization: BGRImage


@dataclass(frozen=True, slots=True)
class VisualizerConfig:
    step: int = 10
    scale: float = 10


type Visualizer = Callable[[BGRImage, Flow, VisualizerConfig], BGRImage]


@dataclass(frozen=True, slots=True)
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


@dataclass(frozen=True, slots=True)
class DISConfig:
    """Configuration for DIS optical flow."""

    preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST


@dataclass(frozen=True, slots=True)
class SparseToDenseConfig:
    """Configuration for sparse-to-dense optical flow."""

    grid_step: int = 8
    k: int = 128
    sigma: float = 0.05
    use_post_proc: bool = True
    fgs_lambda: float = 500.0
    fgs_sigma: float = 1.5


@dataclass(frozen=True, slots=True)
class EmptyConfig: ...


type FlowAlgorithmConfig = FarnebackConfig | DISConfig | SparseToDenseConfig | NoneType
type FlowAlgorithm[C: FlowAlgorithmConfig] = Callable[[GrayImage, GrayImage, C], Flow]


@overload
def parse_NDArray[DType: np.generic](
    image: NDArray[Any],
    dtype: type[DType],
) -> NDArray[DType]: ...
@overload
def parse_NDArray[Shape: tuple[Any, ...], DType: np.generic](
    image: np.ndarray[Shape, Any],
    dtype: type[DType],
) -> np.ndarray[Shape, np.dtype[DType]]: ...
def parse_NDArray[Shape: tuple[Any, ...], DType: np.generic](
    image: np.ndarray[Shape, Any],
    dtype: type[DType],
) -> np.ndarray[Shape, np.dtype[DType]]:
    if image.dtype != np.dtype(dtype):
        raise TypeError(f"Expected {dtype}, got {image.dtype}")

    return image


def RGB2BGR(image: RGBImage) -> BGRImage:
    return BGRImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.uint8))


def BGR2RGB(image: BGRImage) -> RGBImage:
    return RGBImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.uint8))


def BGR2gray(image: BGRImage) -> GrayImage:
    return GrayImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.uint8))


def RGB2gray(image: RGBImage) -> GrayImage:
    return GrayImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), np.uint8))


def gray2BGR(image: GrayImage) -> BGRImage:
    return BGRImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), np.uint8))
