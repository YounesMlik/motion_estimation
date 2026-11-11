
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .flow import FarnebackConfig, Flow, farneback
from .video import Image, read_video, to_grayscale
from .visualization import BGRImage, colorize


@dataclass(slots=True)
class FlowResult:
    frame: Image
    flow: Flow
    visualization: BGRImage


def process_video(
    path: str,
    scale: float = 0.75,
    max_motion: float = 20.0,
    config: FarnebackConfig | None = None,
) -> Iterator[FlowResult]:
    """Yield optical-flow results for consecutive video frames."""
    if config is None:
        config = FarnebackConfig()

    previous: NDArray[np.uint8] | None = None

    for frame in read_video(path, scale):
        current = to_grayscale(frame)

        if previous is not None:
            flow = farneback(previous, current, config)

            yield FlowResult(
                frame=frame,
                flow=flow,
                visualization=colorize(flow, max_motion),
            )

        previous = current
