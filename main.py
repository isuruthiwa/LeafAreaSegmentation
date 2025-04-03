import os

from predict_leaf_area_index import LeafAreaIndexCalculator

lf = LeafAreaIndexCalculator()

# Directory where 50 images are stored
image_directory = '/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/Plant-Images/'
# List all image file paths
image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if
               img.endswith('front_Color.png')]
#
# for i in range(5):
#     test_image = image_files[30 + i]
#     lf.predictLeafAreaIndex(test_image)

lf.evalGPRModel()