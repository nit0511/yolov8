from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from PIL import Image

model = YOLO('runs/detect/train31/weights/best.pt')




image_dir = "well.v8i.yolov8/test/images"
output_dir = "inference/well/images"

os.makedirs(output_dir, exist_ok=True)


# Iterate through files in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        img_path = os.path.join(image_dir, filename)
        print(filename)

        results = model.predict(source=[img_path], conf=0.25)

        # Show the results
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.show()  # show image
            output_path = os.path.join(output_dir, filename)
            im.save(output_path)

# for calculatin mAP use the below code in terminal
# yolo task=detect mode=val model="runs/detect/train2/weights/best.pt" data="AquariumDS_CLAHE/data.yaml"

