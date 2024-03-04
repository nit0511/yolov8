import torch.cuda
from ultralytics import YOLO
print(torch.cuda.is_available())
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"