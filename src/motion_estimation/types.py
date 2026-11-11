from typing import Any

import numpy as np
from numpy.typing import NDArray


def parse_NDArray[DType: np.generic](
    image: NDArray[Any],
    dtype: type[DType],
) -> NDArray[DType]:
    if image.dtype != np.dtype(dtype):
        raise TypeError(f"Expected {dtype}, got {image.dtype}")

    return image
