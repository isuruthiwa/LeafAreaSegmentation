import os

import gpytorch
import pandas as pd
import torch
from gpytorch.kernels import ArcKernel, PolynomialKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from torch import tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F


from segment_leaf_area_model import SegmentLeafAreaUsingYoloSAM2

# Custom dataset for images
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Define the feature extractor (e.g., a simple neural network)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(102400, 1000),  # Compress to 1000 dimensions
            nn.ReLU(),
            nn.Linear(1000, 50),  # Final output: 50 dimensions
        )

    def forward(self, x):
        return self.net(x)


# GP Model
class DKLGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.feature_extractor = SimpleFeatureExtractor()

        # self.covar_module = ArcKernel(base_kernel=gpytorch.kernels.SpectralDeltaKernel(active_dims=10, num_deltas=30, num_dims=1),
        #                               angle_prior=gpytorch.priors.GammaPrior(0.5, 1),
        #                               radius_prior=gpytorch.priors.GammaPrior(0.5, 2)
        #                               )
        self.covar_module = gpytorch.kernels.SpectralDeltaKernel(active_dims=10, num_deltas=20, num_dims=1)


    def forward(self, x):
        print("L1 = " + str(x.shape))
        features = self.feature_extractor(x)
        print("features = " + str(features.shape))
        mean_x = self.mean_module(features)
        print("mean_x = " + str(mean_x))
        covar_x = self.covar_module(features)
        print("covar_x = " + str(covar_x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define the GP Model
class LAIGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LAIGPModel, self).__init__(train_x, train_y, likelihood)

        # Define a constant mean for simplicity
        self.mean_module = ConstantMean()

        # Use an RBF Kernel (Radial Basis Function)
        # base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=102400)


        self.covar_module = ArcKernel(base_kernel=PolynomialKernel(4),
                                      angle_prior=gpytorch.priors.GammaPrior(2, 2.5),
                                      radius_prior=gpytorch.priors.GammaPrior(2, 2.5)
                                      )
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #         ArcKernel(base_kernel,
        #                   angle_prior=gpytorch.priors.GammaPrior(0.5, 1),
        #                   radius_prior=gpytorch.priors.GammaPrior(3, 2),
        #                   ard_num_dims=102400))


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SimpleFeatureExtractor(nn.Module):
    def __init__(self):
        super(SimpleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(40 * 40, 128)

    def forward(self, x):
        # Debugging shape
        print(f"Input shape before Conv2d: {x.shape}")

        # # Remove extra dimensions if necessary
        # if x.dim() == 5 and x.shape[2] == 1:
        #     x = x.squeeze(2)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class LeafAreaIndexCalculator:

    segment_model = None
    leaf_area_model = None
    gpr_likelihood = None

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
    batch_size = 30

    def trainGPRModel(self):
        # Load the plant images and LAI values
        global gpr, likelihood
        # Store loss values
        losses = []

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

                # Desired fixed size (3x3 in this case)
                fixed_size = (320, 320)

                # Resize the 2D vector

                # Resizing the 2D vector while maintaining aspect ratio
                resized_vector = resize(cropped_mask, fixed_size, anti_aliasing=True)

                image_tensors.append(tensor(resized_vector, dtype=torch.float32))
                plant_id = image_file.split('/')[-1].split('_')[0]
                leaf_area_index = df.loc[df['Plant_number'] == plant_id, 'LAI'].values[0]
                leaf_area_index_tensors.append(tensor(leaf_area_index, dtype=torch.float32))

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            image_tensors = torch.stack(image_tensors)
            leaf_area_index_tensors = torch.stack(leaf_area_index_tensors)

            # Reshape or flatten image_tensors
            # image_tensors = image_tensors.view(image_tensors.size(0), -1)  # [batch_size, num_features]

            # Ensure that leaf_area_index_tensors is of correct shape
            # leaf_area_index_tensors = leaf_area_index_tensors.view(-1)  # [batch_size]

            print(image_tensors.shape)
            print(leaf_area_index_tensors.shape)

            # Dataset and DataLoader
            dataset = ImageDataset(image_tensors, leaf_area_index_tensors)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr = DKLGP(image_tensors, leaf_area_index_tensors, likelihood)

            # Training the model
            gpr.train()
            likelihood.train()

            # Use an optimizer, e.g., Adam
            optimizer = torch.optim.Adam(gpr.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)

            # # Training loop (one image at a time)
            # for i, (image, label) in enumerate(dataloader):
            #     print(image)
            #     print(label)
            #     # image = image.unsqueeze(0)  # Add batch and channel dimensions
            #     # Update the model with the current image and label
            #     gpr.set_train_data(inputs=image, targets=label, strict=True)
            #
            #     # Zero gradients from the previous step
            #     optimizer.zero_grad()
            #
            #     # Forward pass
            #     output = gpr(image)
            #
            #     # Compute loss
            #     loss = -mll(output, label)
            #     loss.backward()
            #
            #     # Take a step
            #     optimizer.step()
            #
            #     print(f"Processed image {i + 1}/{len(dataloader)} with loss: {loss.item()}")
            #     print(output)
            #     losses.append(loss.item())  # Record loss value
            #
            # print("Training completed!")

            # TEST CODE
            # gpr.eval()
            # likelihood.eval()

            # val_image = image_files[34]
            #
            # plant_id = val_image.split('/')[-1].split('_')[0]
            # leaf_area_index = df.loc[df['Plant_number'] == plant_id, 'LAI'].values[0]
            # val_leaf_area_index = tensor([leaf_area_index], dtype=torch.float32)
            #
            # # Desired fixed size (3x3 in this case)
            # fixed_size = (320, 320)
            # val_mask, val_cropped_mask = self.segment_model.predict(val_image, 1)
            #
            # # Resizing the 2D vector while maintaining aspect ratio
            # resized_vector = resize(val_cropped_mask, fixed_size, anti_aliasing=True)
            # resized_tensor = tensor(resized_vector, dtype=torch.float32)
            #
            # # resized_tensor = resized_tensor.unsqueeze(0)
            # print("Tensor = " + str(resized_tensor.shape))

            # Reshape or flatten image_tensors
            # resized_tensor = resized_tensor.view(resized_tensor.size(0), -1)
            #
            # with torch.no_grad():
            #     val_predictions = gpr(resized_tensor)
            #     mean_predictions = val_predictions.mean  # Mean of the GP posterior
            #     print("val_pred = " + str(val_predictions))
            #     print("val = " + str(val_leaf_area_index))
            #     print("mean_pred = " + str(mean_predictions))
            # # END TEST CODE

            # Training loop
            for j in range(self.batch_size):
                optimizer.zero_grad()
                output = gpr(image_tensors)
                loss = -mll(output, leaf_area_index_tensors)
                print(loss)
                # Reduce output to a scalar
                scalar_output = loss.sum()
                # Backward pass
                scalar_output.backward()
                # loss.backward()
                optimizer.step()
                print(f'Iteration {j + 1}/{self.batch_size} - Loss: {scalar_output.item()}')
                losses.append(scalar_output.item())  # Record loss value

            # Save the model and likelihood
            torch.save({
                'model_state_dict': gpr.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'train_inputs': gpr.train_inputs,
                'train_targets': gpr.train_targets
            }, 'gpytorch_model.pth')

            # Plot the Loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, self.batch_size + 1), losses, label="Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Negative Log Marginal Likelihood (Loss)")
            plt.title("Training Loss of GPR Model")
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig("training_loss.png", format='png', dpi=1024, bbox_inches='tight')

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
            loaded_model = DKLGP(checkpoint['train_inputs'], checkpoint['train_targets'], loaded_likelihood)
            loaded_model.load_state_dict(checkpoint['model_state_dict'])

            # Set the model and likelihood to evaluation mode
            loaded_model.eval()
            loaded_likelihood.eval()

            self.leaf_area_model = loaded_model
            self.gpr_likelihood =  loaded_likelihood

        return self.leaf_area_model

    def predictLeafAreaIndex(self, image_under_test):
        leaf_area_index_model = self.loadGPRModel()

        # Desired fixed size (3x3 in this case)
        fixed_size = (320, 320)
        mask, cropped_mask = self.segment_model.predict(image_under_test, 0)

        # Resizing the 2D vector while maintaining aspect ratio
        resized_vector = resize(cropped_mask, fixed_size, anti_aliasing=True)
        resized_tensor = tensor(resized_vector.reshape(1, -1), dtype=torch.float32)

        # Reshape or flatten image_tensors
        resized_tensor = resized_tensor.view(resized_tensor.size(0), -1)

        # Predictions with uncertainty
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predicted_LAI = leaf_area_index_model(resized_tensor)
            mean = predicted_LAI.mean
            lower, upper = predicted_LAI.confidence_region()

            return predicted_LAI.mean.item()

