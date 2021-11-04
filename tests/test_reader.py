import os

import pytest
from car_seal.custom_vision.reader import read_and_resize_image


@pytest.mark.parametrize(
    "image_path, max_byte_size",
    [],
)
def test_read_and_resize_image(image_path, max_byte_size):
    image_bytes = read_and_resize_image(
        image_path=image_path, max_byte_size=max_byte_size
    )
    assert len(image_bytes) < max_byte_size
