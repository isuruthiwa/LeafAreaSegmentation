import os
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from ultralytics import YOLO

from utils.helper_functions import show_mask, crop_image_from_bounding_box


class SegmentLeafAreaUsingYoloSAM2:
    yolo_model = None
    sam2_model = None

    # Checkpoint paths and model type definitions
    DEVICE = "cuda"
    SAM2_CHECKPOINT_PATH = "utils/SAM_Model_Weights/sam2_hiera_tiny.pt"
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
        self.sam2_model = self.getSAMModel()

    def getLeafAreaBoundingBox(self, image_path, padding=0):
        results = self.yolo_model(image_path)
        bounding_box = np.array(results[0].boxes[0].xyxy[0].cpu())
        #Add a padding around the detected bounding box to increase the accuracy
        bounding_box[0] -= padding
        bounding_box[1] -= padding
        bounding_box[2] += padding
        bounding_box[3] += padding
        return bounding_box

    def getSAMModel(self):
        if self.sam2_model is None:
            self.sam2_model = build_sam2(self.MODEL_CFG, self.SAM2_CHECKPOINT_PATH, device=self.DEVICE)
        return self.sam2_model

    def segmentLeafAreaFromSAM2(self, image, bounding_box, plot_segmentation=0):
        sam2_model = self.getSAMModel()
        predictor = SAM2ImagePredictor(sam2_model)
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bounding_box,
            multimask_output=False)

        best_mask = masks[scores.argmax()]
        best_score = scores[scores.argmax()]

        show_mask(image, plot_segmentation, bounding_box, best_mask, best_score, self.output_dir, self.file_name)

        cropped_mask = crop_image_from_bounding_box(best_mask, bounding_box)
        return best_mask, cropped_mask

    def predict(self, image_path, plot_prediction=0):
        image = cv2.imread(image_path)
        bounding_box = self.getLeafAreaBoundingBox(image_path, 25)
        return self.segmentLeafAreaFromSAM2(image, bounding_box, plot_prediction)
