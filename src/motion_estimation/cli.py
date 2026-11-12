import argparse
from collections.abc import Iterable

import cv2
import numpy as np

from .pipeline import process_video
from .types import FlowResult


def render_frame(flow_result: FlowResult):
    combined = np.hstack(
        (
            flow_result.frame,
            flow_result.visualization,
        )
    )
    cv2.imshow("Original | Optical Flow", combined)


def render_video(flow_results: Iterable[FlowResult]) -> None:
    try:
        for flow_result in flow_results:
            render_frame(flow_result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def run(path: str) -> None:
    """Run the optical-flow demo with an OpenCV display."""
    try:
        flow = process_video(path)
        render_video(flow)
    except KeyboardInterrupt:
        print("Exiting program")


def main() -> None:
    """Parse command-line arguments and launch the demo."""

    parser = argparse.ArgumentParser(description="Dense optical-flow visualization")

    parser.add_argument("video")
    args = parser.parse_args()

    run(args.video)
