from ultralytics import YOLO
import os

# Specify the save directory for training runs
save_dir = 'model_runs'
os.makedirs(save_dir, exist_ok=True)

model = YOLO("../utils/YOLO_Model_Weights/yolo11n.pt")
model.train(data="data.yaml", epochs=20, project=save_dir, name="results")
