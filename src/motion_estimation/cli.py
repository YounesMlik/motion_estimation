
import argparse

import cv2
import numpy as np

from .pipeline import process_video


def run(
    path: str,
    scale: float = 0.75,
    max_motion: float = 20.0,
) -> None:
    """Run the optical-flow demo with an OpenCV display."""

    try:
        for result in process_video(path, scale, max_motion):
            combined = np.hstack((
                result.frame,
                result.visualization,
            ))

            cv2.imshow("Original | Optical Flow", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    """Parse command-line arguments and launch the demo."""

    parser = argparse.ArgumentParser(
        description="Dense optical-flow visualization"
    )

    parser.add_argument("video")
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--max-motion", type=float, default=20.0)

    args = parser.parse_args()

    run(args.video, args.scale, args.max_motion)
