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
# from utils.perturbation_levels_single import PERTURBATION_LEVELS
from mnist_helper_functions import *

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

# for future reference, the training dataset
trainset = datasets.MNIST(f'{current_dir}/data/', train=True, download=False, transform=transform)
# Not shuffling because we want to retrieve the image by index if necessary
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)

# Load the test data
# data path current_dir plus 'data/'
datapath = os.path.join(current_dir, 'data/')
testset = datasets.MNIST(datapath, train=False, download=False, transform=transform)
# Not shuffling because we want to retrieve the image by index if necessary
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# # Test the model on the test dataset
# correct = 0
# total = 0

# # Store Softmax output for each image prediction
# num_columns = 12
# data_np = np.empty((0, num_columns))

# # .no_grad() disables gradient computation, which reduces memory usage
# with torch.no_grad():
#     for inputs, labels in testloader:
#         outputs_original = net(inputs)
#         # convert log probabilities to probabilities
#         outputs = torch.exp(outputs_original)
#         _, predicted = torch.max(outputs.data, 1) # outputs.shape
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
        
#         outputs_np = outputs.data.numpy()
#         labels_np = labels.numpy().reshape(-1, 1)
#         predicted_np = predicted.numpy().reshape(-1, 1)
#         combined_np = np.hstack((outputs_np, labels_np, predicted_np))

#         data_np = np.vstack((data_np, combined_np))

# # Print the accuracy of the model on the test dataset
# accuracy = 100 * correct / total # 98.38
# print('Accuracy on test dataset: %.2f%%' % accuracy)

# # sanity check
# # Extracting the 11th and 12th columns (indexed as 10 and 11)
# labels = data_np[:, 10]
# predictions = data_np[:, 11]

# # Comparing the two columns to find where values are the same and where they are different
# same_values_mask = labels == predictions

# # Summing the values that are the same and different
# same_values_count = np.sum(same_values_mask)
# different_values_count = np.sum(~same_values_mask)

# # Calculating and printing the percentage of values that are the same
# percentage_same = (same_values_count / data_np.shape[0]) * 100
# print(f"Accuracy on test dataset: {percentage_same:.2f}%")

# Making sure the labels and predictions are as expected
# np.unique(data_np[:, 10])
# array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
# np.unique(data_np[:, 11])
# array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

pertubation_filepath = 'scripts/data/MNIST/raw/t1210k-perturbation-levels-idx0-ubyte'
test_np = save_predictions_perturbed(net, testloader, pertubation_filepath, True)
train_np = save_predictions(net, trainloader, True)
# find the indices of the misclassified images
# unequal_indices, unequal_rows = find_unequal_rows(test_np, 10, 11)
# display the misclassified images
conf_matrix1, conf_matrix2 = display_misclassifications_side_by_side(train_np, test_np, 10, 11, 20, 7)
# get the number of misclassifications
train_misclassifications = sum_misclassifications(conf_matrix1)
test_misclassifications = sum_misclassifications(conf_matrix2)
# plot the misclassifications on training dataset and test dataset
# plot_misclassifications(train_misclassifications, test_misclassifications, True)
# print(conf_matrix1)
# Generate the k-means centroids
correct_preds = train_np[train_np[:, 10] == train_np[:, 11]]
centroids, cluster_labels = get_kmeans_centroids(correct_preds, debug=True)
# correct training predictions to centroids
train_match_condition = train_np[:, 10] == train_np[:, 11]
train_correct_predictions = train_np[train_match_condition]
correct_train_distances = mean_distance_to_centroids(train_correct_predictions, centroids)

# incorrect training predictions to centroids
train_mismatch_condition = train_np[:, 10] != train_np[:, 11]
train_incorrect_predictions = train_np[train_mismatch_condition]
incorrect_train_distances = mean_distance_to_centroids(train_incorrect_predictions, centroids)

# alternatively
#correct_train_distances_v2 = calculate_distances_to_centroids(train_np, equal=True, debug=False)

#################################
# Same data for testing dataset #
#################################
test_correct_preds = test_np[test_np[:, 10] == test_np[:, 11]]
test_centroids, testing_cluster_labels = get_kmeans_centroids(test_correct_preds, debug=True)
# correct testing predictions to centroids
test_match_condition = test_np[:, 10] == test_np[:, 11]
test_correct_predictions = test_np[test_match_condition]
correct_test_distances = mean_distance_to_centroids(test_correct_predictions, test_centroids)

# incorrect testing predictions to centroids
test_mismatch_condition = test_np[:, 10] != test_np[:, 11]
test_incorrect_predictions = test_np[test_mismatch_condition]
incorrect_test_distances = mean_distance_to_centroids(test_incorrect_predictions, test_centroids)

# Plot mean correct prediction distances to centroids side by side, for training and testing
plot_mean_distances_x2(correct_train_distances, correct_test_distances, predictions_type="Correct")

#############################################################################
# Same data for training and testing dataset, but incorrect classifications #
#############################################################################

incorrect_train_distances = mean_distance_to_centroids(train_incorrect_predictions, centroids)
incorrect_test_distances = mean_distance_to_centroids(test_incorrect_predictions, test_centroids)
plot_mean_distances_x2(incorrect_train_distances, incorrect_test_distances, predictions_type="Incorrect")


# Plot accuracy vs distance to centroids
train_class_accuracies = calculate_class_accuracies(train_np)
test_class_accuracies = calculate_class_accuracies(test_np)

# plot_accuracy_vs_distance(correct_train_distances, train_class_accuracies)

plot_mean_distances_x2(correct_train_distances, incorrect_train_distances, correct_test_distances, incorrect_test_distances)

# plot mean distances double bar chart
plot_mean_distances_double_bars(correct_train_distances, incorrect_train_distances, correct_test_distances, incorrect_test_distances, save=True)

# Plot accuracy vs distance to centroids, with linear fit
plot_accuracy_vs_distance_linear_fit(correct_train_distances, train_class_accuracies, correct_test_distances, test_class_accuracies)

# boxplots of distances to centroids for training dataset
d2c_train_correct = calculate_distances_to_centroids(train_correct_predictions, centroids)
d2c_train_incorrect = calculate_distances_to_centroids(train_incorrect_predictions, centroids)

# # boxplots of distances to centroids for testing dataset
d2c_test_correct = calculate_distances_to_centroids(test_correct_predictions, test_centroids)
d2c_test_incorrect = calculate_distances_to_centroids(test_incorrect_predictions, test_centroids)

# boxplots alt function - THE GOOD ONE
boxplots_side_by_side_x2(d2c_train_correct, d2c_train_incorrect, d2c_test_correct, d2c_test_incorrect, False, True)

# same boxplots, using only the training dataset correct class prediction centroids
# # boxplots of distances to centroids for testing dataset
d2c_test_correct_train_centroids = calculate_distances_to_centroids(test_correct_predictions, centroids)
d2c_test_incorrect_train_centroids = calculate_distances_to_centroids(test_incorrect_predictions, centroids)

# boxplots alt function - with testing data using training centroids
boxplots_side_by_side_x2(d2c_train_correct, d2c_train_incorrect, d2c_test_correct_train_centroids, d2c_test_incorrect_train_centroids, False, True, title1="Training Data Boxplots of Softmax Distances to Training Centroids", title2="Testing Data Boxplots of Softmax Distances to Training Centroids")

# bar charts of distances to centroids for training dataset
plot_digit_averages(train_correct_predictions, train_incorrect_predictions, color1='skyblue', color2='lightcoral', data="MNIST Training Data")

# bar charts of distances to centroids for testing dataset
plot_digit_averages(test_correct_predictions, test_incorrect_predictions, color1='lightgreen', color2='lightcoral', data="MNIST Testing Data")

# data on overlap between distances to centroids for correct and incorrect predictions
# centroid_distance_overlap_plain_text(d2c_train_correct, d2c_train_incorrect, d2c_test_correct, d2c_test_incorrect)

centroid_distance_overlap_latex(d2c_train_correct, d2c_train_incorrect, d2c_test_correct, d2c_test_incorrect)
# statistical significance tests
lowest_values = find_lowest_values(d2c_train_incorrect)
accuracy_results = calculate_accuracy_decrements(d2c_test_correct, d2c_test_incorrect, lowest_values)
overall_results = calculate_accuracy_decrements_overall(d2c_test_correct, d2c_test_incorrect, lowest_values)
plot_accuracy_decrements(accuracy_results, overall_results)

single_plot_accuracy_decrements(accuracy_results, overall_results, save=True)

# STOPPED HERE
# Next steps:
# 1. Determine the accuracy of the model on the Training dataset given only examples from a threshold distance from the centroids
# 2. Determine the accuracy of the model on the Test dataset given only examples from a threshold distance from the centroids
# 3. Train 10 models on the training dataset, using softmax outputs as inputs, and the labels as targets, where given a digit, 
# the model predicts if the prediction is correct or not
# 4. Repeat step 3 for the test dataset
# 5. Based on the two metrics, distance to centroid and predictions, finish paper.
