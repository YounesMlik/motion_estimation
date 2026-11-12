from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray

from .flow import farneback
from .types import (
    FlowAlgorithm,
    FlowAlgorithmConfig,
    FlowResult,
    Visualizer,
    VisualizerConfig,
)
from .video import BGR_to_grayscale, read_video
from .visualization import draw_flow_vectors


def process_video(
    path: str,
    *,
    visualizer: Visualizer = draw_flow_vectors,
    visualizer_config: VisualizerConfig | None = None,
    flow_algorirtm: FlowAlgorithm = farneback,
    flow_config: FlowAlgorithmConfig | None = None,
) -> Generator[FlowResult]:
    """Yield optical-flow results for consecutive video frames."""

    if visualizer_config is None:
        visualizer_config = VisualizerConfig()

    previous: NDArray[np.uint8] | None = None

    for frame in read_video(path):
        current = BGR_to_grayscale(frame)

        if previous is not None:
            flow = flow_algorirtm(previous, current, flow_config)

            yield FlowResult(
                frame=frame,
                flow=flow,
                visualization=visualizer(frame, flow, visualizer_config),
            )

        previous = current
