from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from motion_estimation.types import BGR2gray, BGRImage, GrayImage

type Image = NDArray[np.uint8]
type Mask = NDArray[np.bool_]
type Samples = NDArray[
    np.uint8
]  # (N, H, W, C) internally, with grayscale represented using an explicit singleton channel.

type Update = tuple[
    np.ndarray,  # sample indices
    np.ndarray,  # destination rows
    np.ndarray,  # destination columns
    np.ndarray,  # values
]

OFFSETS = np.array(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]
)


@dataclass(frozen=True, slots=True)
class ViBeConfig:
    nb_samples: int = 20
    req_matches: int = 2
    radius: float = 20.0
    subsampling_factor: int = 16

    blink_increment: int = 15
    blink_decrement: int = 1
    blink_max: int = 150
    blink_threshold: int = 50

    def __post_init__(self) -> None:
        if self.nb_samples < 1:
            raise ValueError("nb_samples must be positive")
        if not 1 <= self.req_matches <= self.nb_samples:
            raise ValueError("req_matches must be between 1 and nb_samples")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.subsampling_factor <= 1:
            raise ValueError("subsampling_factor must be positive")
        if self.blink_increment < 1:
            raise ValueError("blink_increment must be positive")
        if self.blink_decrement < 1:
            raise ValueError("blink_decrement must be positive")
        if self.blink_max < 1:
            raise ValueError("blink_max must be positive")
        if self.blink_max > 255:
            raise ValueError("blink_max must not exceed 255")
        if not 1 <= self.blink_threshold <= self.blink_max:
            raise ValueError("blink_threshold must be between 1 and blink_max")


class ViBe:
    def __init__(self, config: ViBeConfig | None, *, seed: int | None = None):
        self.config = config if config is not None else ViBeConfig()
        self.rng = np.random.default_rng(seed)
        self.shape: tuple = None  # pyright: ignore[reportAttributeAccessIssue]
        self.samples = None  # (S, Y, X)
        self.blink_count = None
        self.previous_foreground = None

    def initialize(self, frame: GrayImage):
        h, w = frame.shape
        ys, xs = np.indices(frame.shape)

        self.samples = np.empty((self.config.nb_samples, h, w), dtype=np.uint8)
        self.shape = frame.shape
        self.blink_count = np.zeros(self.shape, dtype=np.uint8)

        for i in range(self.config.nb_samples):
            dy, dx = OFFSETS[self.rng.integers(8, size=self.shape)].transpose(2, 0, 1)
            self.samples[i] = frame[
                np.clip(ys + dy, 0, h - 1),
                np.clip(xs + dx, 0, w - 1),
            ]

    def apply(self, frame: GrayImage):
        self._validate_frame(frame)
        if self.samples is None:
            raise RuntimeError("Call initialize() first")

        foreground = self._classify(frame, self.samples)
        temporal, spatial = self._update(frame, foreground)
        self._apply_update(temporal)
        self._apply_update(spatial)

        return foreground.astype(np.uint8) * 255

    def _validate_frame(self, frame: GrayImage):
        if self.samples is None:
            raise RuntimeError("Call initialize() first")

        if frame.shape != self.shape:
            raise ValueError("Frame shape changed")

    def _classify(self, frame: GrayImage, samples: np.ndarray):
        cfg = self.config

        diff = np.maximum(samples, frame) - np.minimum(samples, frame)
        matches = np.count_nonzero(diff < cfg.radius, axis=0)

        return matches < cfg.req_matches

    def _update(
        self, frame: GrayImage, foreground: np.ndarray
    ) -> tuple[Update, Update]:
        blinking = self._detect_blinking(foreground)
        background = ~foreground & ~blinking

        temporal = self._temporal_updates(frame, background)
        spatial = self._spatial_updates(frame, background)

        return temporal, spatial

    def _detect_blinking(self, foreground: np.ndarray):
        if self.previous_foreground is None:
            self.previous_foreground = foreground.copy()
            return np.zeros(self.shape, dtype=bool)

        if self.blink_count is None:
            raise RuntimeError("Blink counter not initialized")

        cfg = self.config
        changed = foreground != self.previous_foreground
        count = self.blink_count

        self.blink_count = np.where(
            changed,
            np.minimum(count + cfg.blink_increment, cfg.blink_max),
            np.maximum(count, cfg.blink_decrement) - cfg.blink_decrement,
        ).astype(np.uint8)
        self.previous_foreground = foreground.copy()

        return self.blink_count >= cfg.blink_threshold

    def _temporal_updates(self, frame: GrayImage, background: np.ndarray) -> Update:
        cfg = self.config
        ys, xs = np.indices(self.shape)

        update = background & (
            self.rng.integers(cfg.subsampling_factor, size=self.shape) == 0
        )

        indices = self.rng.integers(cfg.nb_samples, size=self.shape)

        return indices[update], ys[update], xs[update], frame[update]

    def _spatial_updates(self, frame: GrayImage, background: np.ndarray) -> Update:
        cfg = self.config
        h, w = self.shape
        ys, xs = np.indices(self.shape)

        update = background & (
            self.rng.integers(cfg.subsampling_factor, size=self.shape) == 0
        )

        dy, dx = OFFSETS[self.rng.integers(8, size=self.shape)].transpose(2, 0, 1)
        ny, nx = ys + dy, xs + dx

        valid = update & (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)

        indices = self.rng.integers(cfg.nb_samples, size=self.shape)

        return indices[valid], ny[valid], nx[valid], frame[valid]

    def _apply_update(self, update: Update):
        if self.samples is None:
            raise RuntimeError("Call initialize() first")

        indices, ys, xs, values = update
        self.samples[indices, ys, xs] = values

    def reset(self):
        self.samples = None
        self.shape = None  # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def process_video(
        video: Iterable[BGRImage],
        config: ViBeConfig | None = None,
        seed: int | None = None,
    ) -> Iterable[Mask]:
        vibe = ViBe(config, seed=seed)

        return _process_video(video, vibe)


def _process_video(video: Iterable[BGRImage], vibe) -> Iterable[Mask]:
    for i, frame in enumerate(video):
        frame = BGR2gray(frame)
        if i == 0:
            vibe.initialize(frame)
            continue

        mask = vibe.apply(frame)
        yield mask
