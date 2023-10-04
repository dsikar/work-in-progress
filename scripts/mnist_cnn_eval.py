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
from utils.perturbation_levels import PERTURBATION_LEVELS

for i in range(0, len(PERTURBATION_LEVELS['gaussian_noise'])):
  print(PERTURBATION_LEVELS['gaussian_noise'][i][0], PERTURBATION_LEVELS['gaussian_noise'][i][1])   

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

# our evaluate function
def evaluate_perturbed_images(testloader):
    # Evaluate the model on the test dataset for different values of noise
    pt = Perturbation()
    # Instanciate the DistanceMetric class
    dm = DistanceMetric(num_channels=1, num_bins=50, val_range=(-1,1))

    for value in np.arange(0.0, 1.1, 0.1):
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
                    inputs[i] = pt.add_shot_noise(inputs[i].squeeze(0), value)
                    # calculate distances
                    bd += dm.BhattacharyaDistance(inputs_copy[i].squeeze().numpy(), inputs[i].squeeze().numpy())
                    kl += dm.KLDivergence(inputs_copy[i].squeeze().numpy(), inputs[i].squeeze().numpy())
                    hi += dm.HistogramIntersection(inputs_copy[i].squeeze().numpy(), inputs[i].squeeze().numpy())
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute and print the accuracy on the test dataset for the current "value"
        accuracy = 100 * correct / total
        bd /= total
        kl /= total
        hi /= total
        print('Accuracy with noise value = %.1f: %.2f%%, Bhattacharyya Distance: %.4f, KL Divergence: %.4f, Histogram Intersection: %.4f' % (value, accuracy, bd, kl, hi))
        #print('Accuracy with noise value = %.1f: %.2f%%' % (value, accuracy))
        # STOPPED HERE, next, add values to dictionary, go through every value of noise, and plot the results
        
# Accuracy on test dataset: 98.38%
# Accuracy with noise value = 0.0: 98.38%, Bhattacharyya Distance: -0.0000, KL Divergence: 0.0000, Histogram Intersection: 1.0000
# Accuracy with noise value = 0.1: 97.80%, Bhattacharyya Distance: 0.0136, KL Divergence: 0.1134, Histogram Intersection: 0.9123
# Accuracy with noise value = 0.2: 95.07%, Bhattacharyya Distance: 0.0279, KL Divergence: 0.2260, Histogram Intersection: 0.8346
# Accuracy with noise value = 0.3: 87.23%, Bhattacharyya Distance: 0.0416, KL Divergence: 0.3407, Histogram Intersection: 0.7647
# Accuracy with noise value = 0.4: 74.26%, Bhattacharyya Distance: 0.0552, KL Divergence: 0.4630, Histogram Intersection: 0.7009
# Accuracy with noise value = 0.5: 60.58%, Bhattacharyya Distance: 0.0684, KL Divergence: 0.5876, Histogram Intersection: 0.6434
# Accuracy with noise value = 0.6: 46.77%, Bhattacharyya Distance: 0.0812, KL Divergence: 0.7162, Histogram Intersection: 0.5917
# Accuracy with noise value = 0.7: 34.91%, Bhattacharyya Distance: 0.0941, KL Divergence: 0.8536, Histogram Intersection: 0.5444
# Accuracy with noise value = 0.8: 25.45%, Bhattacharyya Distance: 0.1067, KL Divergence: 0.9904, Histogram Intersection: 0.5016
# Accuracy with noise value = 0.9: 19.35%, Bhattacharyya Distance: 0.1191, KL Divergence: 1.1309, Histogram Intersection: 0.4630
# Accuracy with noise value = 1.0: 15.77%, Bhattacharyya Distance: 0.1314, KL Divergence: 1.2774, Histogram Intersection: 0.4279      

# Create a network instance
net = Net()

# Load the saved model from a file
# PATH = 'mnist_vanilla_cnn_hyperion_20230426110500.pth' # trained on hyperion, Accuracy on test dataset: 98.35%
PATH = 'models/mnist_vanilla_cnn_local_202306241859.pth' # trained on google colab, 
# prepend current_dir to PATH
PATH = os.path.join(current_dir, PATH)
net = Net()
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
evaluate_perturbed_images(testloader)

# # with modified inputs
# correct = 0
# total = 0
# pt = Perturbation()
# with torch.no_grad():
#     for inputs, labels in testloader:
#         # clone the batch of inputs
#         inputs_original = inputs.clone()
#         # loop through each image in the cloned batch
#         for i in range(inputs.shape[0]):
#             inputs[i] = pt.add_shot_noise(inputs[i].squeeze(0), 0.05)
#             inputs[i] = inputs[i].unsqueeze(0)
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# Print the accuracy of the model on the test dataset
# accuracy = 100 * correct / total
# print('Accuracy on test dataset: %.2f%%' % accuracy)



# # Evaluate the model on the test dataset for different values of "value"
# # for value in np.arange(2, -2.1, -0.1):
# #     # Reset the accuracy counters
# #     correct = 0
# #     total = 0

# #     # Test the model on the test dataset with added "value"
# #     with torch.no_grad():
# #         for inputs, labels in testloader:
# #             inputs += value
# #             inputs = torch.clamp(inputs, -1, 1)  # clip values to range -1 to 1
# #             outputs = net(inputs)
# #             _, predicted = torch.max(outputs.data, 1)
# #             total += labels.size(0)
# #             correct += (predicted == labels).sum().item()

# #     # Compute and print the accuracy on the test dataset for the current "value"
# #     accuracy = 100 * correct / total
# #     print('Accuracy with shifted value = %.1f: %.2f%%' % (value, accuracy))

# # Add a variable to every array in the inputs variable, and print the accuracy and histogram overlap
# for value in np.arange(1, -1.1, -0.1):
#     with torch.no_grad():
#         correct = 0
#         incorrect = 0
#         total = 0
#         overlap = 0
#         imgs_total = 0

#         for inputs, labels in testloader:
#             inputs_copy = inputs.clone()
#             inputs += value
#             inputs = torch.clamp(inputs, -1, 1)  # clip values to range -1 to 1
#             outputs = net(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#             # Compute the histogram overlap between the original and modified arrays
#             #original_array = inputs.numpy().flatten()
#             #modified_array = (inputs - value).numpy().flatten()
#             #overlap += histogram_overlap(original_array, modified_array)                   
            
#             # loop through each image in the batch
#             for i in range(inputs.shape[0]):
#                 original_array = inputs_copy[i].squeeze().numpy() #.numpy()inputs[i].squeeze().flatten()
#                 modified_array = (inputs[i]).squeeze().numpy() #.numpy().flatten()
#                 # clamp the modified array to the range -1 to 1
#                 #modified_array = np.clip(modified_array, -1, 1)
#                 # compute the histogram intersection between the original and modified arrays               
#                 overlap += dm.HistogramIntersection(original_array, modified_array)
#             # increment the testloader_total counter
#             imgs_total += inputs.shape[0]

#         accuracy = 100 * correct / total
#         overlap /= imgs_total 
#         print('Shift value: %.1f, Accuracy: %.2f%%, Overlap: %.4f' % (value, accuracy, overlap))

# # Accuracy on test dataset: 99.17%
# # Shift value: 1.0, Accuracy: 87.51%, Overlap: 0.1363
# # Shift value: 0.9, Accuracy: 93.63%, Overlap: 0.0172
# # Shift value: 0.8, Accuracy: 97.25%, Overlap: 0.0239
# # Shift value: 0.7, Accuracy: 98.42%, Overlap: 0.0319
# # Shift value: 0.6, Accuracy: 98.81%, Overlap: 0.0405
# # Shift value: 0.5, Accuracy: 99.01%, Overlap: 0.0509
# # Shift value: 0.4, Accuracy: 99.21%, Overlap: 0.0591
# # Shift value: 0.3, Accuracy: 99.24%, Overlap: 0.0691
# # Shift value: 0.2, Accuracy: 99.25%, Overlap: 0.0810
# # Shift value: 0.1, Accuracy: 99.20%, Overlap: 0.0957
# # Shift value: 0.0, Accuracy: 99.17%, Overlap: 1.0000
# # Shift value: -0.1, Accuracy: 99.14%, Overlap: 0.0962
# # Shift value: -0.2, Accuracy: 99.09%, Overlap: 0.0816
# # Shift value: -0.3, Accuracy: 99.01%, Overlap: 0.0702
# # Shift value: -0.4, Accuracy: 98.90%, Overlap: 0.0599
# # Shift value: -0.5, Accuracy: 98.76%, Overlap: 0.0521
# # Shift value: -0.6, Accuracy: 98.45%, Overlap: 0.0414
# # Shift value: -0.7, Accuracy: 97.87%, Overlap: 0.0320
# # Shift value: -0.8, Accuracy: 96.98%, Overlap: 0.0243
# # Shift value: -0.9, Accuracy: 95.49%, Overlap: 0.0179
# # Shift value: -1.0, Accuracy: 92.94%, Overlap: 0.0766

# # 50 bins
# # Accuracy on test dataset: 98.38%
# # Shift value: 1.0, Accuracy: 49.01%, Overlap: 0.9197
# # Shift value: 0.9, Accuracy: 63.03%, Overlap: 0.9200
# # Shift value: 0.8, Accuracy: 74.65%, Overlap: 0.9227
# # Shift value: 0.7, Accuracy: 82.12%, Overlap: 0.9253
# # Shift value: 0.6, Accuracy: 90.94%, Overlap: 0.9267
# # Shift value: 0.5, Accuracy: 95.67%, Overlap: 0.9301
# # Shift value: 0.4, Accuracy: 97.61%, Overlap: 0.9328
# # Shift value: 0.3, Accuracy: 98.17%, Overlap: 0.9359
# # Shift value: 0.2, Accuracy: 98.45%, Overlap: 0.9398
# # Shift value: 0.1, Accuracy: 98.52%, Overlap: 0.9482
# # Shift value: 0.0, Accuracy: 98.38%, Overlap: 1.0000
# # Shift value: -0.1, Accuracy: 98.38%, Overlap: 0.9483
# # Shift value: -0.2, Accuracy: 98.21%, Overlap: 0.9403
# # Shift value: -0.3, Accuracy: 97.96%, Overlap: 0.9360
# # Shift value: -0.4, Accuracy: 97.68%, Overlap: 0.9335
# # Shift value: -0.5, Accuracy: 97.06%, Overlap: 0.9298
# # Shift value: -0.6, Accuracy: 95.95%, Overlap: 0.9271
# # Shift value: -0.7, Accuracy: 94.27%, Overlap: 0.9246
# # Shift value: -0.8, Accuracy: 91.74%, Overlap: 0.9232
# # Shift value: -0.9, Accuracy: 87.53%, Overlap: 0.9065
# # Shift value: -1.0, Accuracy: 81.18%, Overlap: 0.9056