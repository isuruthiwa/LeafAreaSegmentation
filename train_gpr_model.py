import os

import cv2
import gpytorch
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, morphology, feature
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler, RobustScaler


from segment_leaf_area_model import SegmentLeafAreaUsingYoloSAM2

batch_size = 30
segment_model = SegmentLeafAreaUsingYoloSAM2()

df = pd.read_csv('/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/PlantData_Updated.csv')
num_batches = df.size // batch_size

# Directory where your 50 images are stored
image_directory = '/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/Plant-Images/'

# List all image file paths (assuming you have 50 images in the directory)
image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if
               img.endswith('front_Color.png')]

image_tensors = list()
leaf_area_index_tensors = list()


# def extract_features(mask_tensor):
#     mask = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
#     features = []
#
#     # Area
#     leaf_area = np.count_nonzero(mask)
#     features.append(leaf_area)
#
#     # Bounding Box
#     y_indices, x_indices = np.nonzero(mask)
#     if len(y_indices) == 0:
#         # Return zero vector if mask is empty
#         return torch.zeros(10, dtype=torch.float32)
#
#     min_y, max_y = np.min(y_indices), np.max(y_indices)
#     min_x, max_x = np.min(x_indices), np.max(x_indices)
#     bbox_area = (max_y - min_y + 1) * (max_x - min_x + 1)
#     features.append(bbox_area)
#
#     # Density
#     density = leaf_area / bbox_area if bbox_area > 0 else 0
#     features.append(density)
#
#     # Contour (needed for shape descriptors)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return torch.zeros(10, dtype=torch.float32)
#
#     cnt = max(contours, key=cv2.contourArea)
#
#     # Perimeter
#     perimeter = cv2.arcLength(cnt, True)
#     features.append(perimeter)
#
#     # Compactness: P² / A
#     compactness = (perimeter ** 2) / leaf_area if leaf_area > 0 else 0
#     features.append(compactness)
#
#     # Aspect ratio
#     x, y, w, h = cv2.boundingRect(cnt)
#     aspect_ratio = w / h if h > 0 else 0
#     features.append(aspect_ratio)
#
#     # Extent: Area / Bounding box area
#     extent = leaf_area / (w * h) if w * h > 0 else 0
#     features.append(extent)
#
#     # Solidity: Area / Convex hull area
#     hull = cv2.convexHull(cnt)
#     hull_area = cv2.contourArea(hull)
#     solidity = leaf_area / hull_area if hull_area > 0 else 0
#     features.append(solidity)
#
#     # Hu Moments (shape descriptors)
#     moments = cv2.moments(mask)
#     hu = cv2.HuMoments(moments).flatten()
#     log_hu1 = -np.sign(hu[0]) * np.log10(abs(hu[0])) if hu[0] != 0 else 0
#     features.append(log_hu1)  # Add just the 1st Hu moment as example
#
#     return torch.tensor(features, dtype=torch.float32)


# def extract_features(mask_tensor):
#     """Robust feature extraction for leaf area index calculation"""
#     if np.count_nonzero(mask) == 0:
#         return torch.zeros(25, dtype=torch.float32)
#
#     try:
#         # Clean mask and get region
#         mask_clean = ndimage.binary_fill_holes(mask)
#         labeled = measure.label(mask_clean)
#         regions = measure.regionprops(labeled)
#
#         if not regions:
#             return torch.zeros(25, dtype=torch.float32)
#
#         region = max(regions, key=lambda x: x.area)
#         y0, x0 = region.centroid
#         orientation = region.orientation
#
#         # Crop the region
#         minr, minc, maxr, maxc = region.bbox
#         image_crop = image[minr:maxr, minc:maxc]
#         mask_crop = mask_clean[minr:maxr, minc:maxc]
#
#         masked_rgb = image_crop.copy()
#         masked_rgb[~mask_crop.astype(bool)] = 0
#
#         # Extract channels
#         r, g, b = masked_rgb[..., 0], masked_rgb[..., 1], masked_rgb[..., 2]
#         valid_pixels = mask_crop.astype(bool)
#
#         # Basic shape
#         area = region.area
#         perimeter = region.perimeter
#         convex_area = region.convex_area
#         compactness = (perimeter ** 2) / area if area > 0 else 0
#         convexity = area / convex_area if convex_area > 0 else 0
#
#         # Color stats
#         mean_r = r[valid_pixels].mean()
#         mean_g = g[valid_pixels].mean()
#         mean_b = b[valid_pixels].mean()
#         std_r = r[valid_pixels].std()
#         std_g = g[valid_pixels].std()
#         std_b = b[valid_pixels].std()
#
#         # NDVI-style greenness index
#         greenness_index = (2 * g.astype(np.float32) - r - b)[valid_pixels].mean()
#
#         # Haralick from grayscale
#         gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
#         glcm = feature.graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
#         contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
#
#         feature_dict = {
#             'area': area,
#             'perimeter': perimeter,
#             'compactness': compactness,
#             'convexity': convexity,
#             # 'orientation_sin': np.sin(orientation),
#             # 'orientation_cos': np.cos(orientation),
#             'mean_r': mean_r, 'mean_g': mean_g, 'mean_b': mean_b,
#             'std_r': std_r, 'std_g': std_g, 'std_b': std_b,
#             'greenness_index': greenness_index,
#             'haralick_contrast': contrast
#         }
#
#         features = np.array(list(feature_dict.values()), dtype=np.float32)
#         features = np.nan_to_num(features)
#
#         # Pad or trim to 25 dimensions
#         if features.shape[0] < 25:
#             features = np.pad(features, (0, 25 - features.shape[0]))
#         elif features.shape[0] > 25:
#             features = features[:25]
#
#         return torch.tensor(features, dtype=torch.float32)
#
#     except Exception as e:
#         print(f"Feature extraction error: {e}")
#         return torch.zeros(25, dtype=torch.float32)

def extract_features(mask_tensor):
    """Robust feature extraction for leaf area index calculation"""
    # Move to CPU and convert to numpy
    image_tensor = mask_tensor.detach().cpu()

    # Convert torch tensor to numpy array in (H, W, C) format
    if image_tensor.ndim == 3:
        if image_tensor.shape[0] == 3:  # (C, H, W)
            image_np = image_tensor.permute(1, 2, 0).numpy()
        elif image_tensor.shape[2] == 3:  # (H, W, C)
            image_np = image_tensor.numpy()
        else:
            raise ValueError("Input tensor must be shape (3,H,W) or (H,W,3)")
    else:
        raise ValueError("Expected 3D image tensor (C,H,W) or (H,W,C)")

    # Convert to grayscale and extract mask
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Clean mask
    mask = (binary_mask > 0).astype(np.uint8)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    # Label connected components
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)

    if not regions:
        return torch.zeros(10, dtype=torch.float32)  # Return zero vector if no leaf found

    # Use largest region
    region = max(regions, key=lambda x: x.area)

    area = region.area
    perimeter = region.perimeter
    convex_area = region.convex_area
    eccentricity = region.eccentricity
    extent = region.extent
    solidity = region.solidity
    orientation = region.orientation
    bbox_area = region.bbox_area
    compactness = (perimeter ** 2) / area if area > 0 else 0
    convexity = area / convex_area if convex_area > 0 else 0

    features = [
        area,
        perimeter,
        convex_area,
        eccentricity,
        extent,
        solidity,
        # np.sin(orientation),
        # np.cos(orientation),
        compactness,
        convexity
    ]

    features = np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.tensor(features, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = RobustScaler()

# Flatten the image tensors before stacking
for image_file in image_files:
    mask, cropped_mask = segment_model.predict(image_file, plot_prediction=0)
    mask_bool = mask.astype(bool)

    image = cv2.imread(image_file)
    # 1. Pixelwise masking: Keep original shape, black out background
    masked_image = image.copy()
    masked_image[~mask_bool] = 0  # Set background to black

    with torch.no_grad():
        masked_image = torch.tensor(np.array(masked_image)).to(device)
        feature_tensor = extract_features(masked_image)
    image_tensors.append(feature_tensor)

    plant_id = image_file.split('/')[-1].split('_')[0]
    leaf_area_index = df.loc[df['Plant_number'] == plant_id, 'LAI'].values[0]

    # leaf_area_index_scaled = scaler.fit_transform(leaf_area_index.reshape(-1, 1)) # y = LAI values
    leaf_area_index_tensors.append(torch.tensor(leaf_area_index, dtype=torch.float32))

train_n = 40

X = np.array(leaf_area_index_tensors).reshape(-1, 1)  # Ensure 2D shape
leaf_area_index_scaled = scaler.fit_transform(X) # y = LAI values

leaf_area_index_scaled = leaf_area_index_scaled.squeeze()
x_scaler = StandardScaler()

train_x = torch.stack(image_tensors[:train_n]).to(device, dtype=torch.float32)
train_y = torch.tensor(np.array(leaf_area_index_scaled[:train_n]), dtype=torch.float32).to(device, dtype=torch.float32)

test_x = torch.stack(image_tensors[train_n:]).to(device, dtype=torch.float32)
test_y = torch.tensor(np.array(leaf_area_index_scaled[train_n:]), dtype=torch.float32).to(device, dtype=torch.float32)

train_x = torch.tensor(x_scaler.fit_transform(train_x.to(dtype=torch.float32).cpu().numpy()), dtype=torch.float32).to(device, dtype=torch.float32)
test_x = torch.tensor(x_scaler.transform(test_x.to(dtype=torch.float32).cpu().numpy())).to(device, dtype=torch.float32)

print("-------Train Y-------")
print(train_y.dtype)
print(test_y.dtype)
print("-------Train X-------")
print(train_x.dtype)
print(test_x.dtype)
print("--------------")

print(train_x)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.AdditiveKernel(gpytorch.kernels.ArcKernel(base_kernel=gpytorch.kernels.MaternKernel(1.5),
        #                               angle_prior=gpytorch.priors.GammaPrior(1, 2.5),
        #                               radius_prior=gpytorch.priors.GammaPrior(1, 2.5)
        #                               ) + gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=21))
        # self.covar_module = gpytorch.kernels.ArcKernel(base_kernel=gpytorch.kernels.MaternKernel(1.5),
        #                               angle_prior=gpytorch.priors.GammaPrior(1, 2.5),
        #                               radius_prior=gpytorch.priors.GammaPrior(1, 2.5)
        #                               ) + gpytorch.kernels.MaternKernel(nu=1.5)
        # self.covar_module = gpytorch.kernels.ScaleKernel(self.cov_mod)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=8000))
        self.covar_module = (
                gpytorch.kernels.RBFKernel(active_dims=[0, 1])  # Spatial dimensions
                * gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[2])  # Feature dimension
                + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())  # Long-range variations
                # + gpytorch.kernels.PolynomialKernel(2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

if torch.cuda.is_available():
    train_y = train_y.to(device,  dtype=torch.float32)
    test_y = test_y.to(device,  dtype=torch.float32)
    test_x = test_x.to(device,  dtype=torch.float32)
    train_x = train_x.to(device, dtype=torch.float32)

model = GPRegressionModel(train_x, train_y, likelihood).to(device)

training_iterations = 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        # iterator.set_postfix(loss=loss.item())
        optimizer.step()

        if i % 20 == 0:
            print(f"[{i}] Loss: {loss.item():.4f}")

train()

# Switch to evaluation mode
model.eval()
likelihood.eval()

# Get predictions for the single sample
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    val_predictions = likelihood(model(test_x))

# Extract the mean and variance of the predictions
mean_predictions = val_predictions.mean  # Mean of the GP posterior
variance_predictions = val_predictions.variance  # Variance of the GP posterior

# Print predictions and ground truth
print("Mean predictions:", mean_predictions)
mean_predictions_cp = mean_predictions.cpu().numpy().reshape(-1, 1)
mp_sc = scaler.inverse_transform(mean_predictions_cp)
print("Mean predictions:", mp_sc)
print("Variance predictions:", variance_predictions)
print("Ground truth:", X[train_n:])

# Optionally, calculate evaluation metrics (e.g., Mean Squared Error)
mse = np.mean((mean_predictions_cp - X[train_n:]) ** 2).item()

print("Mean Squared Error (MSE):", mse)
