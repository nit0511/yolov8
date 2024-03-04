import torch.cuda
from ultralytics import YOLO
print(torch.cuda.is_available())
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    # Display model information (optional)
    model.info()

    #model.train(data='AquariumDS_CLAHE/data.yaml', epochs=250, imgsz=640, plots=True)
    model.train(data='DIAT Aquarium.v4i.yolov8/data.yaml', epochs=250, imgsz=640, plots=True, device=0,)


