import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from utils.distance_utils import histogram_overlap

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
PATH = 'mnist_vanilla_cnn_hyperion_20230508082100.pth' # trained on google colab, 
net = Net()
net.load_state_dict(torch.load(PATH))

# Load the test data
testset = datasets.MNIST('data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Test the model on the test dataset
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the accuracy of the model on the test dataset
accuracy = 100 * correct / total
print('Accuracy on test dataset: %.2f%%' % accuracy)

# Evaluate the model on the test dataset for different values of "value"
# for value in np.arange(2, -2.1, -0.1):
#     # Reset the accuracy counters
#     correct = 0
#     total = 0

#     # Test the model on the test dataset with added "value"
#     with torch.no_grad():
#         for inputs, labels in testloader:
#             inputs += value
#             inputs = torch.clamp(inputs, -1, 1)  # clip values to range -1 to 1
#             outputs = net(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     # Compute and print the accuracy on the test dataset for the current "value"
#     accuracy = 100 * correct / total
#     print('Accuracy with shifted value = %.1f: %.2f%%' % (value, accuracy))

# Add a variable to every array in the inputs variable, and print the accuracy and histogram overlap
for value in np.arange(1, -1.1, -0.1):
    with torch.no_grad():
        correct = 0
        incorrect = 0
        total = 0
        overlap = 0

        for inputs, labels in testloader:
            inputs += value
            inputs = torch.clamp(inputs, -1, 1)  # clip values to range -1 to 1
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute the histogram overlap between the original and modified arrays
            original_array = inputs.numpy().flatten()
            modified_array = (inputs - value).numpy().flatten()
            overlap += histogram_overlap(original_array, modified_array)                   

        accuracy = 100 * correct / total
        overlap /= len(testloader) 
        print('Shift value: %.1f, Accuracy: %.2f%%, Overlap: %.4f' % (value, accuracy, overlap))

# Accuracy on test dataset: 99.17%
# Shift value: 1.0, Accuracy: 87.51%, Overlap: 0.1363
# Shift value: 0.9, Accuracy: 93.63%, Overlap: 0.0172
# Shift value: 0.8, Accuracy: 97.25%, Overlap: 0.0239
# Shift value: 0.7, Accuracy: 98.42%, Overlap: 0.0319
# Shift value: 0.6, Accuracy: 98.81%, Overlap: 0.0405
# Shift value: 0.5, Accuracy: 99.01%, Overlap: 0.0509
# Shift value: 0.4, Accuracy: 99.21%, Overlap: 0.0591
# Shift value: 0.3, Accuracy: 99.24%, Overlap: 0.0691
# Shift value: 0.2, Accuracy: 99.25%, Overlap: 0.0810
# Shift value: 0.1, Accuracy: 99.20%, Overlap: 0.0957
# Shift value: 0.0, Accuracy: 99.17%, Overlap: 1.0000
# Shift value: -0.1, Accuracy: 99.14%, Overlap: 0.0962
# Shift value: -0.2, Accuracy: 99.09%, Overlap: 0.0816
# Shift value: -0.3, Accuracy: 99.01%, Overlap: 0.0702
# Shift value: -0.4, Accuracy: 98.90%, Overlap: 0.0599
# Shift value: -0.5, Accuracy: 98.76%, Overlap: 0.0521
# Shift value: -0.6, Accuracy: 98.45%, Overlap: 0.0414
# Shift value: -0.7, Accuracy: 97.87%, Overlap: 0.0320
# Shift value: -0.8, Accuracy: 96.98%, Overlap: 0.0243
# Shift value: -0.9, Accuracy: 95.49%, Overlap: 0.0179
# Shift value: -1.0, Accuracy: 92.94%, Overlap: 0.0766