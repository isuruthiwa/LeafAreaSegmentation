import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from utils.helper_functions import show_mask


class SegmentLeafAreaUsingYoloSAM2:
    yolo_model = None
    sam2_model = None

    # Checkpoint paths and model type definitions
    DEVICE = "cuda"
    SAM2_CHECKPOINT_PATH = "utils/SAM_Model_Weights/SAM2/sam2_hiera_tiny.pt"
    MODEL_CFG = "sam2_hiera_t.yaml"
    YOLO_MODEL_CHECKPOINT = "train_yolo_model/model_runs/results/weights/best.pt"

    # Specify the output directory and file name
    output_dir = 'output_results'
    file_name = 'plot_image.png'
    os.makedirs(output_dir, exist_ok=True)

    # use bfloat16 for the run
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    def __init__(self):
        super().__init__()
        self.yolo_model = YOLO(self.YOLO_MODEL_CHECKPOINT)
        self.sam2_model = build_sam2(self.MODEL_CFG, self.SAM2_CHECKPOINT_PATH, device=self.DEVICE)

    def getLeafAreaBoundingBox(self, image_path):
        results = self.yolo_model(image_path)
        return np.array(results[0].boxes[0].xyxy[0].cpu())

    def segmentLeafAreaFromSAM2(self, image, bounding_box, plot_segmentation=0):
        sam2_model = build_sam2(self.MODEL_CFG, self.SAM2_CHECKPOINT_PATH, device=self.DEVICE)
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bounding_box,
            multimask_output=False)

        best_mask = masks[scores.argmax()]
        best_score = scores[scores.argmax()]

        if plot_segmentation:
            show_mask(image, best_mask, best_score, self.output_dir, self.file_name)

        return best_mask, best_score

    def predict(self, image_path, plot_prediction=0):
        image = cv2.imread(image_path)
        bounding_box = self.getLeafAreaBoundingBox(image_path)
        return self.segmentLeafAreaFromSAM2(image, bounding_box, plot_prediction)
