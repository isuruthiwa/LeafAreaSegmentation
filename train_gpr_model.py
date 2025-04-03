import os
import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.kernels import ArcKernel, PolynomialKernel, RBFKernel
from gpytorch.kernels.keops import MaternKernel
from matplotlib import pyplot as plt
from skimage.transform import resize
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LargeFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()

        # Convolutional Layers for 2D Input (320x320 images)
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 1, 320, 320]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [batch_size, 16, 160, 160]

            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 16, 160, 160]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [batch_size, 32, 80, 80]

            torch.nn.Conv2d(32, 40, kernel_size=3, stride=1, padding=1),  # Input: [batch_size, 32, 80, 80]
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # Output: [batch_size, 40, 40, 40]
        )

        # Fully Connected Layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(40 * 40, 1000),  # Input:
            torch.nn.ReLU(),
            # torch.nn.Linear(1000, 500)
        )

    def forward(self, x):
        print("Input shape to feature extractor:", x.shape)  # Debugging line
        x = self.conv_layers(x)
        print("Shape after convolutional layers:", x.shape)  # Debugging line
        x = x.view(x.size(0), -1)  # Flatten the tensor
        print("Shape after flattening:", x.shape)  # Debugging line
        x = self.fc_layers(x)
        print("Shape after fully connected layers:", x.shape)  # Debugging line
        x = x.view(-1)  # Flatten the tensor
        print("Shape after flattening:", x.shape)  # Debugging line
        return x

feature_extractor = LargeFeatureExtractor().to(device)
scaler = RobustScaler()

# Flatten the image tensors before stacking
for image_file in image_files:
    mask, cropped_mask = segment_model.predict(image_file)
    plt.imshow(mask)
    plt.show()
    fixed_size = (320, 320)
    resized_vector = resize(cropped_mask, fixed_size, anti_aliasing=True)

    with torch.no_grad():
        resized_tensor = torch.tensor(np.array(resized_vector)).unsqueeze(0).to(device)
        feature_tensor = feature_extractor(resized_tensor)
    image_tensors.append(feature_tensor)

    plant_id = image_file.split('/')[-1].split('_')[0]
    leaf_area_index = df.loc[df['Plant_number'] == plant_id, 'LAI'].values[0]

    # leaf_area_index_scaled = scaler.fit_transform(leaf_area_index.reshape(-1, 1)) # y = LAI values
    leaf_area_index_tensors.append(torch.tensor(leaf_area_index, dtype=torch.float32))

train_n = 40

X = np.array(leaf_area_index_tensors).reshape(-1, 1)  # Ensure 2D shape
leaf_area_index_scaled = scaler.fit_transform(X) # y = LAI values

print(leaf_area_index_scaled)
leaf_area_index_scaled = leaf_area_index_scaled.squeeze()

train_x = torch.stack(image_tensors[:train_n])
train_y = torch.tensor(np.array(leaf_area_index_scaled[:train_n]), dtype=torch.float32)

test_x = torch.stack(image_tensors[train_n:])
test_y = torch.tensor(np.array(leaf_area_index_scaled[train_n:]), dtype=torch.float32)

print(train_y.shape)
print(test_y.shape)
print(gpytorch.__version__)


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.AdditiveKernel(ArcKernel(base_kernel=MaternKernel(1.5),
        #                               angle_prior=gpytorch.priors.GammaPrior(1, 2.5),
        #                               radius_prior=gpytorch.priors.GammaPrior(1, 2.5)
        #                               ) + gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=8000))
        self.cov_mod = ArcKernel(base_kernel=MaternKernel(1.5),
                                      angle_prior=gpytorch.priors.GammaPrior(1, 2.5),
                                      radius_prior=gpytorch.priors.GammaPrior(1, 2.5)
                                      ) + gpytorch.kernels.MaternKernel(nu=1.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.cov_mod)
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=8000))

    def forward(self, x):
        print("GP Regression shape : "+ str(x.shape))
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

if torch.cuda.is_available():
    train_y = train_y.to(device,  dtype=torch.float32)
    test_y = test_y.to(device,  dtype=torch.float32)
    test_x = test_x.to(device,  dtype=torch.float32)

model = GPRegressionModel(train_x, train_y, likelihood).to(device)

training_iterations = 1

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
        # print("Train_x shape = " + str(train_x.shape))
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()

        print(loss)

train()

# Switch to evaluation mode
model.eval()
likelihood.eval()


# Print the shape of the first test sample
print("Shape of the first test sample:", test_x.shape)

# Get predictions for the single sample
with torch.no_grad():
    print("Type of test_x:", test_x.type())
    val_predictions = likelihood(model(test_x))
    print("Model output shape:", val_predictions.mean.shape)
    print(val_predictions)

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
