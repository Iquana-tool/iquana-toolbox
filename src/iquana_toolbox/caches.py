import os
import threading
from functools import lru_cache
from cachetools import TTLCache, LRUCache
import cv2
import numpy as np


@lru_cache(maxsize=128)
def get_image_from_url_cached(url: str) -> np.ndarray:
    # This WILL persist across different requests
    if not os.path.exists(url) or not os.path.isfile(url):
        raise FileNotFoundError(f"Image for {url} not found.")
    img = cv2.imread(url)
    if img is None:
        raise ValueError(f"Image could not be loaded from {url}. Might be unsupported file type.")
    return img


class ImageCacheEntry:
    def __init__(self, image):
        self.image = image  # Image in HWC format (numpy array)
        self.crop = None  # [min_x, min_y, max_x, max_y] in relative coordinates (0 to 1)

    def get_image(self):
        if self.crop is None:
            return self.image
        min_x = int(self.crop[0] * self.image.shape[1])
        min_y = int(self.crop[1] * self.image.shape[0])
        max_x = int(self.crop[2] * self.image.shape[1])
        max_y = int(self.crop[3] * self.image.shape[0])
        return self.image[min_y:max_y, min_x:max_x]

    def set_crop(self, min_x, min_y, max_x, max_y):
        assert 0 <= min_x < max_x <= 1, "Crop coordinates must be between 0 and 1 and min < max."
        assert 0 <= min_y < max_y <= 1, "Crop coordinates must be between 0 and 1 and min < max."
        self.crop = [min_x, min_y, max_x, max_y]

    def unset_crop(self):
        self.crop = None


class ImageCache:
    def __init__(self, max_items=100, ttl_seconds=3600):
        # maxsize is the number of items, ttl is how long they stay (1 hour)
        self.cache = TTLCache(maxsize=max_items, ttl=ttl_seconds)

    def set(self, key, image: np.ndarray):
        # We store the Entry object just like you had it
        self.cache[key] = ImageCacheEntry(image)

    def get(self, key):
        if key not in self.cache:
            raise KeyError(f"Image for {key} expired or not found.")
        return self.cache[key].get_image()

    def set_focused_crop(self, key, min_x, min_y, max_x, max_y):
        if key in self.cache:
            self.cache[key].set_crop(min_x, min_y, max_x, max_y)

    def unset_focused_crop(self, key):
        if key in self.cache:
            self.cache[key].unset_crop()


    def __contains__(self, key):
        return key in self.cache


class ModelCache:
    def __init__(self, size_limit=3):
        self.cache = LRUCache(maxsize=size_limit)
        self.user_to_model_key = {}
        self.lock = threading.Lock()

    def get(self, user_identifier):
        with self.lock:
            # .get() in cachetools automatically updates the "recency"
            # so the most used models stay at the top.
            model = self.cache.get(user_identifier)
            if model is None:
                raise KeyError(f"Model for {user_identifier} not loaded.")
            return model

    def put(self, user_identifier, model_registry_key, model):
        with self.lock:
            # If the cache is full, cachetools automatically
            # discards the Least Recently Used model.
            self.cache[user_identifier] = model
            self.user_to_model_key[user_identifier] = model_registry_key

    def check_if_loaded(self, user_identifier, model_registry_key):
        with self.lock:
            return user_identifier in self.user_to_model_key and self.user_to_model_key[user_identifier] == model_registry_key
