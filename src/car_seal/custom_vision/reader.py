import cv2
import numpy as np


def read_and_resize_image(image_path: str, max_byte_size: int) -> bytes:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    current_byte_size = len(cv2.imencode(".jpg", image)[1].tobytes())
    if current_byte_size > max_byte_size:
        scale_factor: int = int(np.cbrt(max_byte_size / current_byte_size) * 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), scale_factor]
        image_bytes = cv2.imencode(".jpg", image, encode_param)[1].tobytes()
    else:
        image_bytes = cv2.imencode(".jpg", image)[1].tobytes()

    return image_bytes
