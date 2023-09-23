import cv2
import requests
import random
import base64
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
import io


url = "http://0.0.0.0:7000/detect/"

img = cv2.imread("ppe.jpg")
img_str = cv2.imencode(".jpg", img)[1].tobytes().decode("ISO-8859-1")

stream = BytesIO(img_str.encode("ISO-8859-1"))
image = Image.open(stream).convert("RGB")
open_cv_image = np.array(image) 

query = {
    'image': img_str,
    'image_name': 'image3.jpg',
    'camera_id': '123',
    'image_time': '2023-09-04 22:13:23.123456',
    'model_type': 'object detection',
    'model_framework': 'yolov5',
    'model_config': {
    'is_track':True,
    'conf_thres': 0.1,
    'iou_thres': 0.1,
    'max_det': 300,
    'agnostic_nms': True,
    'augment': False,
    
   }}


r = requests.post(url, json=query)
data = r.json()
print(r.json())