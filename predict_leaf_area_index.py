import os

import gpytorch
import pandas as pd
import torch
from gpytorch.kernels import PolynomialKernel, ArcKernel
from gpytorch.means import ConstantMean
from skimage.transform import resize
from torch import tensor
from torchvision import transforms

from segment_leaf_area_model import SegmentLeafAreaUsingYoloSAM2


# Define the GP Model
class LAIGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(LAIGPModel, self).__init__(train_x, train_y, likelihood)

        # Define a constant mean for simplicity
        self.mean_module = ConstantMean()

        # Use an RBF Kernel (Radial Basis Function)
        self.covar_module = ArcKernel(base_kernel=PolynomialKernel(6),
                                      angle_prior=gpytorch.priors.GammaPrior(0.5, 5),
                                      radius_prior=gpytorch.priors.GammaPrior(0.5, 5)
                                      )


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LeafAreaIndexCalculator:

    segment_model = None
    leaf_area_model = None

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
        df = pd.read_csv('/home/isuruthiwa/Research/LAI/Basil_Leaf_Area_Computer_vision/PlantData.csv')
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
                leaf_area_index_tensors.append(tensor([leaf_area_index], dtype=torch.float32))

            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            image_tensors = torch.stack(image_tensors)
            leaf_area_index_tensors = torch.stack(leaf_area_index_tensors)

            # Reshape or flatten image_tensors
            image_tensors = image_tensors.view(image_tensors.size(0), -1)  # [batch_size, num_features]

            # Ensure that leaf_area_index_tensors is of correct shape
            leaf_area_index_tensors = leaf_area_index_tensors.view(-1)  # [batch_size]

            print(image_tensors.shape)
            print(leaf_area_index_tensors.shape)

            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gpr = LAIGPModel(image_tensors, leaf_area_index_tensors, likelihood)

            # Training the model
            gpr.train()
            likelihood.train()

            # Use an optimizer, e.g., Adam
            optimizer = torch.optim.Adam(gpr.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gpr)

            # Training loop
            for j in range(100):
                optimizer.zero_grad()
                output = gpr(image_tensors)
                loss = -mll(output, leaf_area_index_tensors)
                loss.backward()
                optimizer.step()
                print(f'Iteration {j + 1}/100 - Loss: {loss.item()}')

            # Save the model and likelihood
            torch.save({
                'model_state_dict': gpr.state_dict(),
                'likelihood_state_dict': likelihood.state_dict(),
                'train_inputs': gpr.train_inputs,
                'train_targets': gpr.train_targets
            }, 'gpytorch_model.pth')



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

            self.leaf_area_model = loaded_model

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

