import cv2
import numpy as np

from motion_estimation.types import BGRImage, VisualizerConfig, parse_NDArray

from .flow import Flow


def colorize(frame: BGRImage, flow: Flow, config: VisualizerConfig) -> BGRImage:
    """Visualize flow direction as hue and magnitude as brightness."""

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1],
    )

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)

    hsv[..., 0] = (angle * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(
        magnitude * 255,
        0,
        255,
    ).astype(np.uint8)

    return BGRImage(parse_NDArray(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), np.uint8))


def draw_flow_vectors(
    frame: BGRImage,
    flow: Flow,
    config: VisualizerConfig,
) -> BGRImage:
    """Overlay sampled optical-flow vectors on the original frame.

    Args:
        frame: Original BGR image.
        flow: Optical flow with horizontal and vertical displacement.
        step: Distance between sampled vectors in pixels.
        scale: Multiplier applied to vector lengths.

    Returns:
        A copy of the original frame with motion vectors drawn on it.
    """
    if config.step <= 0:
        raise ValueError("step must be positive")

    if config.scale < 0:
        raise ValueError("scale must be non-negative")

    output = frame.copy()
    height, width = flow.shape[:2]

    for y in range(config.step // 2, height, config.step):
        for x in range(config.step // 2, width, config.step):
            dx, dy = flow[y, x]

            if not np.isfinite(dx) or not np.isfinite(dy):
                continue

            start = (x, y)
            end = (
                round(x + float(dx) * config.scale),
                round(y + float(dy) * config.scale),
            )

            cv2.arrowedLine(
                output,
                start,
                end,
                color=(0, 255, 0),
                thickness=1,
                tipLength=0.3,
            )

    # return BGRImage(parse_NDArray(output, np.uint8))
    return BGRImage(parse_NDArray(output, np.uint8))
