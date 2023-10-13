import cv2
import json
import numpy as np
from PIL import Image
import requests

img_path = "assets/rand_cam_test.jpg"
# img = cv2.imread(img_path).tolist()

with open(
    img_path, "rb"
) as image_file:
    response = requests.post(
        "https://wm4uy0lywetgeb-5000.proxy.runpod.net/gangsta_inference",
        files={"image": image_file},
    )