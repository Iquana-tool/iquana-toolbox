from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=128)
def get_image_from_url_cached(url: str):
    # This WILL persist across different requests
    return Image.open(url)
