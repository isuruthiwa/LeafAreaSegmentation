import os

import cv2
import gpytorch
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import feature, measure
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from segment_leaf_area_model import SegmentLeafAreaUsingYoloSAM2

# Move the tensor to the GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Custom dataset for images
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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
        bbox_area,
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


# Define the GP Model
class LAIGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LAIGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = (
                gpytorch.kernels.RBFKernel(active_dims=[0, 1])  # Spatial dimensions
                * gpytorch.kernels.MaternKernel(nu=2.5, active_dims=[2])  # Feature dimension
                + gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())  # Long-range variations
        )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LeafAreaIndexCalculator:

    segment_model = None
    leaf_area_model = None
    gpr_likelihood = None

    y_scaler = RobustScaler()
    x_scaler = StandardScaler()

    def __init__(self):
        super().__init__()
        self.segment_model = SegmentLeafAreaUsingYoloSAM2()

    # Define a transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts image to a PyTorch tensor
    ])

    # Move the tensor to the GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Process in batches (e.g., batch size of 10 to fit into GPU memory)
    batch_size = 50

    def trainGPRModel(self):
        # Load the plant images and LAI values
        global gpr, likelihood
        # Store loss values
        losses = []

        train_n = 30

        df = pd.read_csv('/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/PlantData_Updated.csv')
        num_batches = df.size // self.batch_size

        # Directory where your 50 images are stored
        image_directory = '/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/Plant-Images/'

        # List all image file paths (assuming you have 50 images in the directory)
        image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if
                       img.endswith('front_Color.png')]

        for i in range(1):
            batch_files = image_files[i * self.batch_size: (i + 1) * self.batch_size]
            image_tensors = list()
            leaf_area_index_tensors = list()

            for image_file in batch_files:
                mask, cropped_mask = self.segment_model.predict(image_file)

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
                leaf_area_index_tensors.append(tensor(leaf_area_index, dtype=torch.float32))

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # image_tensors = torch.stack(image_tensors)
            # leaf_area_index_tensors = torch.stack(leaf_area_index_tensors)

            # print(image_tensors.shape)
            # print(leaf_area_index_tensors.shape)

            X = np.array(leaf_area_index_tensors).reshape(-1, 1)  # Ensure 2D shape
            leaf_area_index_scaled = self.y_scaler.fit_transform(X)  # y = LAI values

            leaf_area_index_scaled = leaf_area_index_scaled.squeeze()

            # train_x = image_tensors[:train_n]
            train_x = torch.stack(image_tensors[:train_n]).to(device, dtype=torch.float32)
            train_x = torch.tensor(self.x_scaler.fit_transform(train_x.cpu().numpy()))
            train_y = torch.tensor(np.array(leaf_area_index_scaled[:train_n]), dtype=torch.float32)

            if torch.cuda.is_available():
                train_y = train_y.to(device, dtype=torch.float32)
                train_x = train_x.to(device, dtype=torch.float32)

            # Dataset and DataLoader
            dataset = ImageDataset(image_tensors, leaf_area_index_tensors)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr = LAIGPModel(train_x, train_y, likelihood).to(device)

            # Training the model
            gpr.train()
            likelihood.train()

            # Use an optimizer, e.g., Adam
            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                {'params': gpr.covar_module.parameters()},
                {'params': gpr.mean_module.parameters()},
                {'params': gpr.likelihood.parameters()},
            ], lr=0.01)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)


            print("Training completed!")
            training_iterations = 100

            # Training loop
            iterator = tqdm(range(training_iterations))
            for j in iterator:
                optimizer.zero_grad()
                output = gpr(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                # iterator.set_postfix(loss=loss.item())
                optimizer.step()

                scalar_output = loss.sum()
                losses.append(scalar_output.item())  # Record loss value

                # if i % 20 == 0:
                print(f"[{j}] Loss: {loss.item():.4f}")

            # for j in range(self.batch_size):
            #     optimizer.zero_grad()
            #     output = gpr(train_x)
            #     loss = -mll(output, train_y)
            #     loss.backward()
            #     print(loss)
            #     # Reduce output to a scalar
            #     scalar_output = loss.sum()
            #     # Backward pass
            #     # scalar_output.backward()
            #     # loss.backward()
            #     optimizer.step()
            #     print(f'Iteration {j + 1}/{self.batch_size} - Loss: {scalar_output.item()}')
            #     losses.append(scalar_output.item())  # Record loss value

            # Save the model and likelihood
            torch.save({
                'model_state_dict': gpr.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'train_inputs': gpr.train_inputs,
                'train_targets': gpr.train_targets,
                'x_scaler': self.x_scaler,
                'y_scaler': self.y_scaler
            }, 'gpytorch_model.pth')

            print(losses)
            # Plot the Loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, training_iterations + 1), losses, label="Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Negative Log Marginal Likelihood (Loss)")
            plt.title("Training Loss of GPR Model")
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig("training_loss.png", format='png', dpi=1024, bbox_inches='tight')

            # TEST CODE
            gpr.eval()
            likelihood.eval()

            val_image = image_files[36]

            plant_id = val_image.split('/')[-1].split('_')[0]
            leaf_area_index = df.loc[df['Plant_number'] == plant_id, 'LAI'].values[0]
            val_leaf_area_index = tensor([leaf_area_index], dtype=torch.float32)

            val_mask, val_cropped_mask = self.segment_model.predict(val_image, 1)
            mask_bool = val_mask.astype(bool)
            image = cv2.imread(val_image)
            # 1. Pixelwise masking: Keep original shape, black out background
            masked_image = image.copy()
            masked_image[~mask_bool] = 0  # Set background to black

            with torch.no_grad():
                masked_image = torch.tensor(np.array(masked_image))
                feature_tensor = extract_features(masked_image).unsqueeze(0).to(device)
                print(feature_tensor.size())
                val_predictions = gpr(feature_tensor)
                mean_predictions = val_predictions.mean  # Mean of the GP posterior
                print("val_pred = " + str(val_predictions))
                print("val = " + str(val_leaf_area_index))
                print("mean_pred = " + str(mean_predictions))

                mean_predictions_cp = mean_predictions.cpu().numpy().reshape(-1, 1)
                mp_sc = self.y_scaler.inverse_transform(mean_predictions_cp)

                print("Mean predictions:", mp_sc)
                print("Ground truth:", val_leaf_area_index)

            # END TEST CODE

    def evalGPRModel(self):
        eval_leaf_area_index_model = self.loadGPRModel()
        eval_likelihood = self.gpr_likelihood

        # Directory where 50 images are stored
        image_directory = '/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/Plant-Images/'
        # List all image file paths
        image_files = [os.path.join(image_directory, img) for img in os.listdir(image_directory) if
                       img.endswith('front_Color.png')]
        df = pd.read_csv('/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/PlantData_Updated.csv')

        # Evaluation mode
        eval_leaf_area_index_model.eval()
        eval_likelihood.eval()

        with torch.no_grad():

            for i in range(10):
                val_image = image_files[30 + i]

                plant_id = val_image.split('/')[-1].split('_')[0]
                leaf_area_index = df.loc[df['Plant_number'] == plant_id, 'LAI'].values[0]
                val_leaf_area_index = tensor([leaf_area_index], dtype=torch.float32)

                # Desired fixed size (3x3 in this case)
                fixed_size = (320, 320)
                val_mask, val_cropped_mask = self.segment_model.predict(val_image, 1)



                # Resizing the 2D vector while maintaining aspect ratio
                resized_vector = resize(val_cropped_mask, fixed_size, anti_aliasing=True)
                resized_tensor = tensor(resized_vector, dtype=torch.float32)

                resized_tensor = resized_tensor.unsqueeze(0).unsqueeze(0)
                print("Tensor = " + str(resized_tensor.shape))

                # Reshape or flatten image_tensors
                # resized_tensor = resized_tensor.view(resized_tensor.size(0), -1)

                val_predictions = eval_likelihood(eval_leaf_area_index_model(resized_tensor))
                mean_predictions = val_predictions.mean  # Mean of the GP posterior
                print("val_pred = " + str(val_predictions))
                print("val = " + str(val_leaf_area_index))
                print("mean_pred = " + str(mean_predictions))
                mse = mean_squared_error(val_leaf_area_index.numpy(), [0])

        print(mse)

    def loadGPRModel(self):
        if self.leaf_area_model is None:
            # Load the saved state dictionaries
            checkpoint = torch.load('gpytorch_model.pth')

            # Initialize a new model and likelihood with the same architecture
            loaded_likelihood = gpytorch.likelihoods.GaussianLikelihood()
            loaded_likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            loaded_model = LAIGPModel(checkpoint['train_inputs'], checkpoint['train_targets'], loaded_likelihood)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])

            # Set the model and likelihood to evaluation mode
            loaded_model.eval()
            loaded_likelihood.eval()

            self.leaf_area_model = loaded_model.to(device)
            self.gpr_likelihood =  loaded_likelihood

            # Load scaler
            self.x_scaler = checkpoint['x_scaler']
            self.y_scaler = checkpoint['y_scaler']

        return self.leaf_area_model

    def predictLeafAreaIndex(self, image_under_test):
        leaf_area_index_model = self.loadGPRModel()

        print("Entered predictLeafAreaIndex")

        # Desired fixed size (3x3 in this case)
        mask, cropped_mask = self.segment_model.predict(image_under_test, 0)

        print("Segmentation Done")

        mask_bool = mask.astype(bool)
        image = cv2.imread(image_under_test)
        # 1. Pixelwise masking: Keep original shape, black out background
        masked_image = image.copy()
        masked_image[~mask_bool] = 0  # Set background to black

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            masked_image = torch.tensor(np.array(masked_image)).to(device)
            feature_tensor = extract_features(masked_image).unsqueeze(0)
            print(feature_tensor)
            feature_tensor = torch.tensor(self.x_scaler.transform(feature_tensor.cpu().numpy())).to(device)
            print(feature_tensor)
            predicted_LAI = leaf_area_index_model(feature_tensor)

            print(predicted_LAI.mean)
            mean = predicted_LAI.mean
            lower, upper = predicted_LAI.confidence_region()

            mean_predictions_cp = mean.cpu().numpy().reshape(-1, 1)
            mp_sc = self.y_scaler.inverse_transform(mean_predictions_cp)

            print("Mean predictions:", mp_sc.item())

            return mp_sc.item()

