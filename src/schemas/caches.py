from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=128)
def get_image_from_url_cached(url: str) -> np.ndarray:
    # This WILL persist across different requests
    return cv2.imread(url)
