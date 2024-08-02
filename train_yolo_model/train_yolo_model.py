from ultralytics import YOLO
import os

# Specify the save directory for training runs
save_dir = 'model_runs'
os.makedirs(save_dir, exist_ok=True)

model = YOLO("../utils/YOLO_Model_Weights/yolov8n.pt")
model.train(data="data.yaml", epochs=10, project=save_dir, name="results")
