import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from utils.helper_functions import show_mask

# Checkpoint paths and model type definitions
DEVICE = "cuda"
SAM2_CHECKPOINT_PATH = "utils/SAM_Model_Weights/SAM2/sam2_hiera_tiny.pt"
MODEL_CFG = "sam2_hiera_t.yaml"
YOLO_MODEL_CHECKPOINT = "train_yolo_model/model_runs/results/weights/best.pt"


# use bfloat16 for the run
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()


# Specify the output directory and file name
output_dir = 'output_results'
file_name = 'plot_image.png'
os.makedirs(output_dir, exist_ok=True)

model = YOLO(YOLO_MODEL_CHECKPOINT)

image_path = "/home/isuruthiwa/Research/LAI/Dataset/plant_images/train_images/Plant8_front_Color.png"
results = model(image_path)

image = cv2.imread(image_path)
bounding_box = np.array(results[0].boxes[0].xyxy[0].cpu())

sam2_model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT_PATH, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=bounding_box,
    multimask_output=False)

best_mask = masks[scores.argmax()]
best_score = scores[scores.argmax()]

plt.imshow(image)
show_mask(best_mask, plt.gca())
plt.title(f"Mask, Score: {best_score:.3f}", fontsize=18)

# Save the plot as an image
plt.savefig(f'{output_dir}/{file_name}', format='png', dpi=300)

plt.show()
