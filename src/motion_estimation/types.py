from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NewType

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


Visualizer = Callable[[BGRImage, Flow, VisualizerConfig], BGRImage]


@dataclass(frozen=True, slots=True)
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


FlowAlgorithmConfig = FarnebackConfig
FlowAlgorithm = Callable[[GrayImage, GrayImage, FlowAlgorithmConfig | None], Flow]


def parse_NDArray[DType: np.generic](
    image: NDArray[Any],
    dtype: type[DType],
) -> NDArray[DType]:
    if image.dtype != np.dtype(dtype):
        raise TypeError(f"Expected {dtype}, got {image.dtype}")

    return image


def RGB2BGR(image: RGBImage) -> BGRImage:
    return BGRImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), np.uint8))


def BGR2RGB(image: BGRImage) -> RGBImage:
    return RGBImage(parse_NDArray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), np.uint8))
