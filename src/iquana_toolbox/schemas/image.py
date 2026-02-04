import os
import cv2
from pydantic import BaseModel, field_validator
import base64


class Image(BaseModel):
    """ Model to represent images. """
    id: int
    width: int
    height: int
    color_mode: str
    description: str | None = None
    file_path: str
    thumbnail_file_path: str

    @field_validator("file_path", "thumbnail_file_path")
    def validate_file_path(cls, value):
        if not os.path.exists(value):
            raise FileNotFoundError
        return value

    @classmethod
    def from_db(cls, db_image):
        return cls(
            id=db_image.id,
            width=db_image.width,
            height=db_image.height,
            color_mode=db_image.color_mode,
            description=db_image.description,
            file_path=db_image.file_path,
            thumbnail_file_path=db_image.thumbnail_file_path,
        )

    def load_image(self, as_base64=False):
        """ Load the image from disk. """
        if as_base64:
            with open(self.file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        return cv2.imread(self.file_path)

    def load_thumbnail(self, as_base64=False):
        """ Load the thumbnail image from disk. """
        if as_base64:
            with open(self.thumbnail_file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        return cv2.imread(self.thumbnail_file_path)