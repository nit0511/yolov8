from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from PIL import Image

model = YOLO('runs/detect/train25/weights/best.pt')

img_path = "saved_img/starfish_2.jpg"
output_dir = "saved_img"

results = model.predict(source=[img_path], conf=0.25)

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    output_path = os.path.join(output_dir, "starfish_2_inference.jpg")
    im.save(output_path)