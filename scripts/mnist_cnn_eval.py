import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import sys
import os

# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the module search path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.distance_metrics import DistanceMetric
from utils.perturbations import *
from utils.helper_functions import *
#from utils.perturbation_levels import PERTURBATION_LEVELS
from utils.perturbation_levels_single import PERTURBATION_LEVELS

# for i in range(0, len(PERTURBATION_LEVELS['gaussian_noise'])):
#   print(PERTURBATION_LEVELS['gaussian_noise'][i]['mean'], PERTURBATION_LEVELS['gaussian_noise'][i]['std'])   

# Define transform to normalize data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Our model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Create a network instance
net = Net()
# Load the saved model from a file
# PATH = 'mnist_vanilla_cnn_hyperion_20230426110500.pth' # trained on hyperion, Accuracy on test dataset: 98.35%
PATH = 'models/mnist_vanilla_cnn_local_202306241859.pth' # trained on google colab, 

# prepend current_dir to PATH
PATH = os.path.join(current_dir, PATH)
net.load_state_dict(torch.load(PATH))
# Load the test data
# data path current_dir plus 'data/'
datapath = os.path.join(current_dir, 'data/')
testset = datasets.MNIST(datapath, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Test the model on the test dataset
correct = 0
total = 0

# .no_grad() disables gradient computation, which reduces memory usage
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the accuracy of the model on the test dataset
accuracy = 100 * correct / total # 98.38
print('Accuracy on test dataset: %.2f%%' % accuracy)
# evaluate_perturbed_images(testloader)
# log TODO save original accuracy, KL, BD, HI
data_instance = initialise_data('Vanilla_CNN', PATH, '9d2611b', 'https://github.com/dsikar/ecai2023', 'mnist_cnn_eval.py', accuracy)

# Evaluate the model on the test dataset for different values of noise
pt = Perturbation()
# Note, (-1, 1) is the range of values for the MNIST dataset
dm = DistanceMetric(num_channels=1, num_bins=50, val_range=(-1,1))

# 1. Iterate through the perturbation types
for key in PERTURBATION_LEVELS.keys():
    print("Perturbation type:", key)
    # 2. Iterate through the perturbation parameters
    for k in range(0, len(PERTURBATION_LEVELS[key])):
    # 3. Iterate through the perturbation levels
    #for value in PERTURBATION_LEVELS[key][k]:
        # Reset the accuracy counters
        correct = 0
        total = 0
        # Test the model on the test dataset with added "value"
        with torch.no_grad():
            # running averages for battacharya distance, kl divergence, and histogram intersection
            bd = 0
            kl = 0
            hi = 0
            for inputs, labels in testloader:
                inputs_copy = inputs.clone()
                # loop through each image in the batch
                for i in range(inputs.shape[0]):
                    # Dynamically call the desired perturbation function
                    kwargs = PERTURBATION_LEVELS[key][k]                    
                    # pass numpy array to perturbation function
                    tmp_img = np.array(inputs[i].squeeze().numpy())
                    tmp_img = getattr(pt, key)(tmp_img, **kwargs)
                    # convert back to tensor
                    inputs[i] = torch.from_numpy(tmp_img)
                    # calculate distances
                    bd += dm.BhattacharyaDistance(inputs_copy[i].squeeze().numpy(), inputs[i].squeeze().numpy())
                    kl += dm.KLDivergence(inputs_copy[i].squeeze().numpy(), inputs[i].squeeze().numpy())
                    hi += dm.HistogramIntersection(inputs_copy[i].squeeze().numpy(), inputs[i].squeeze().numpy())
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute and print the accuracy and distances for the current "value"
        accuracy = 100 * correct / total
        bd /= total
        kl /= total
        hi /= total
        print('Accuracy: {:.4f}, with perturbation type {} values = {}, Bhattacharyya Distance: {:.4f}, KL Divergence: {:.4f}, Histogram Intersection: {:.4f}'.format(accuracy, key, kwargs, bd, kl, hi))
        append_results(data_instance, k, bd, kl, hi, key, accuracy)
saved_filename = save_to_pickle(data_instance, "vanilla_cnn_mnist")
print("Saved to file:", saved_filename)