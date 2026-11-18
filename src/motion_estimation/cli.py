import argparse
from collections.abc import Iterable

import cv2
import numpy as np

from motion_estimation.video import show_video

from .pipeline import process_video
from .types import BGRImage, FlowResult, parse_NDArray


def render_flow_result(flow_result: FlowResult) -> BGRImage:
    combined = np.hstack(
        (
            flow_result.frame,
            flow_result.visualization,
        )
    )
    return BGRImage(parse_NDArray(combined, np.uint8))


def run(path: str) -> None:
    """Run the optical-flow demo with an OpenCV display."""
    try:
        flow = process_video(path)
        frames = map(render_flow_result, flow)
        show_video(frames, title="Original | Optical Flow")
    except KeyboardInterrupt:
        print("Exiting program")


def main() -> None:
    """Parse command-line arguments and launch the demo."""

    parser = argparse.ArgumentParser(description="Dense optical-flow visualization")

    parser.add_argument("video")
    args = parser.parse_args()

    run(args.video)
