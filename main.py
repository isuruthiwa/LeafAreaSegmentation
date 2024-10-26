from predict_leaf_area_index import LeafAreaIndexCalculator
from segment_leaf_area_model import SegmentLeafAreaUsingYoloSAM2

# image_path = "/home/isuruthiwa/Research/LAI/Dataset/plant_images/train_images/Plant8_front_Color.png"
# image_path = "/home/isuruthiwa/Research/LAI/Medicinal Dataset/project-1-at-2024-07-24-16-30-6984386a/images/train/32f2855f-4294.jpg"
image_path = "/home/isuruthiwa/Downloads/dbd7e5c5050a3564771dabf6353af714.jpg"
# segment_model = SegmentLeafAreaUsingYoloSAM2()
# mask, cropped_mask = segment_model.predict(image_path,1)

lf = LeafAreaIndexCalculator()
lf.trainGPRModel()