from segment_leaf_area_model import SegmentLeafAreaUsingYoloSAM2

image_path = "/home/isuruthiwa/Research/LAI/Dataset/plant_images/train_images/Plant8_front_Color.png"
segment_model = SegmentLeafAreaUsingYoloSAM2()
mask, score = segment_model.predict(image_path)