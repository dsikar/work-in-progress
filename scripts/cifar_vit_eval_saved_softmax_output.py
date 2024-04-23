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
from utils.perturbation_levels import PERTURBATION_LEVELS
from mnist_helper_functions import *

###################
# CIFAR10 RESULTS #
###################

# Open and load cifar_train_np.npy
linux_path = '/data/CIFAR10/'
windows_path = '\data\CIFAR10\\'
if os.name == 'nt':     
    path = windows_path
else:
    path = linux_path

train_np_logits = np.load(f'{current_dir}{path}cifar_train_np.npy')
train_np = logits_to_softmax(train_np_logits)
sanity_check(train_np)
# Open and load cifar_test_np.npy
test_np_logits = np.load(f'{current_dir}{path}cifar_test_np.npy')
test_np = logits_to_softmax(test_np_logits)
sanity_check(test_np)

# Open and read class_labels.txt
with open(f'{current_dir}{path}class_labels.txt', 'r') as file:
    class_labels = file.read().splitlines()

######################
# CONFUSION MATRICES #
######################
# TODO comment back in
# conf_matrix1, conf_matrix2 = display_misclassifications_side_by_side_cifar10(train_np, test_np, 10, 11, class_labels, 20, 7, True)

# get the number of misclassifications - obsolete
#train_misclassifications = sum_misclassifications(conf_matrix1)
#test_misclassifications = sum_misclassifications(conf_matrix2)

##################################
# TRAINING AND TESTING CENTROIDS #
##################################

# Generate the training dataset k-means centroids for correct predictions 
correct_preds = train_np[train_np[:, 10] == train_np[:, 11]]
centroids, cluster_labels = get_kmeans_centroids(correct_preds, debug=True)

# Generate the testing dataset k-means centroids for correct predictions 
test_correct_preds = test_np[test_np[:, 10] == test_np[:, 11]]
test_centroids, testing_cluster_labels = get_kmeans_centroids(test_correct_preds, debug=True)

###############################################
# TRAINING AND TESTING DISTANCES TO CENTROIDS #
###############################################

# correct training dataset prediction mean distances to centroids
train_match_condition = train_np[:, 10] == train_np[:, 11]
train_correct_predictions = train_np[train_match_condition]
correct_train_distances = mean_distance_to_centroids(train_correct_predictions, centroids)

# incorrect training dataset prediction mean distances to centroids
train_mismatch_condition = train_np[:, 10] != train_np[:, 11]
train_incorrect_predictions = train_np[train_mismatch_condition]
incorrect_train_distances = mean_distance_to_centroids(train_incorrect_predictions, centroids)

# correct testing dataset prediction mean distances to centroids
test_match_condition = test_np[:, 10] == test_np[:, 11]
test_correct_predictions = test_np[test_match_condition]
correct_test_distances = mean_distance_to_centroids(test_correct_predictions, test_centroids)

# incorrect testing dataset prediction mean distances to centroids
test_mismatch_condition = test_np[:, 10] != test_np[:, 11]
test_incorrect_predictions = test_np[test_mismatch_condition]
incorrect_test_distances = mean_distance_to_centroids(test_incorrect_predictions, test_centroids)

##############################################################
# TRAINING AND TESTING MEAN DISTANCE TO CENTROIDS BAR CHARTS # 
##############################################################

# Plot mean correct prediction distances to centroids side by side, for training and testing
# TODO comment back in
# plot_mean_distances_x2(correct_train_distances, incorrect_train_distances, correct_test_distances, incorrect_test_distances)


#plot_mean_distances_x2(incorrect_train_distances, incorrect_test_distances, predictions_type="Incorrect")

# plot mean distances double bar chart
# TODO comment back in
# plot_mean_distances_double_bars(correct_train_distances, incorrect_train_distances, correct_test_distances, incorrect_test_distances, save=True)

# Plot accuracy vs distance to centroids, with linear fit
# TODO comment back in
# plot_accuracy_vs_distance_linear_fit(correct_train_distances, train_class_accuracies, correct_test_distances, test_class_accuracies)

# boxplots of distances to centroids for training dataset
d2c_train_correct = calculate_distances_to_centroids(train_correct_predictions, centroids)
d2c_train_incorrect = calculate_distances_to_centroids(train_incorrect_predictions, centroids)

# # boxplots of distances to centroids for testing dataset
d2c_test_correct = calculate_distances_to_centroids(test_correct_predictions, test_centroids)
d2c_test_incorrect = calculate_distances_to_centroids(test_incorrect_predictions, test_centroids)

# same boxplots, using only the training dataset correct class prediction centroids
# # boxplots of distances to centroids for testing dataset
d2c_test_correct_train_centroids = calculate_distances_to_centroids(test_correct_predictions, centroids)
d2c_test_incorrect_train_centroids = calculate_distances_to_centroids(test_incorrect_predictions, centroids)

boxplots_side_by_side_x2(d2c_train_correct, 
                        d2c_train_incorrect, 
                        d2c_test_correct_train_centroids, 
                        d2c_test_incorrect_train_centroids, 
                        labels=class_labels,
                        save=True, 
                        debug=True,
                        title1='Training Dataset Distribution of Distances to Centroids',
                        title2='Testing Dataset Distribution of Distances to Centroids',
                        xlabel1='CIFAR10 Training Dataset Class Prediction',
                        xlabel2='CIFAR10 Testing Dataset Class Prediction',
                        ylabel1='Distance to Centroid (Logarithmic Scale)',
                        ylabel2='',
                        filename='CIFAR10_boxplots_side_by_side_x2.png',
                        figsize=(20, 5)
                    )

# boxplots alt function - with testing data using training centroids
boxplots_side_by_side_x2(d2c_train_correct, 
                         d2c_train_incorrect, 
                         d2c_test_correct_train_centroids, 
                         d2c_test_incorrect_train_centroids, 
                         save=False, 
                         debug=True, 
                         title1="Training Data Boxplots of Softmax Distances to Training Centroids", 
                         title2="Testing Data Boxplots of Softmax Distances to Training Centroids")

# bar charts of distances to centroids for training dataset
# plot_centroid_distance_bars(train_correct_predictions, train_incorrect_predictions, color1='skyblue', color2='lightcoral', data="CIFAR10 Training Data")
plot_centroid_distance_bars(train_correct_predictions, 
                            train_incorrect_predictions, 
                            labels=class_labels, 
                            color1='lightblue', 
                            color2='lightcoral', 
                            data="CIFAR10 Training Data", 
                            save=True, 
                            filename='CIFAR10_training_plot_centroid_distance_bars.png')

# bar charts of distances to centroids for testing dataset
plot_centroid_distance_bars(test_correct_predictions, 
                            test_incorrect_predictions, 
                            labels=class_labels, 
                            color1='lightgreen', 
                            color2='lightcoral', 
                            data="CIFAR10 Testing Data", 
                            save=True, 
                            filename='CIFAR10_testing_plot_centroid_distance_bars.png')

# data on overlap between distances to centroids for correct and incorrect predictions
#centroid_distance_overlap_plain_text(d2c_train_correct, d2c_train_incorrect, d2c_test_correct, d2c_test_incorrect)

# TODO Set table caption with function argument
# Correct and incorrect classification distance to centroid overlap in LaTeX format
centroid_distance_overlap_latex(d2c_train_correct, 
                                d2c_train_incorrect, 
                                d2c_test_correct, 
                                d2c_test_incorrect, 
                                caption='CIFAR10 centroid distances overlap count for correct predictions above incorrect threshold'
                            )

# STOPPED HERE, Next step, extend y axis on linear plot

# statistical significance tests
lowest_values = find_lowest_values(d2c_train_incorrect)
accuracy_results = calculate_accuracy_decrements(d2c_test_correct, d2c_test_incorrect, lowest_values)
overall_results = calculate_accuracy_decrements_overall(d2c_test_correct, d2c_test_incorrect, lowest_values)
# Testing dataset
# plot_accuracy_decrements(accuracy_results, overall_results)
# plot_accuracy_decrements(accuracy_results, overall_results, labels=class_labels)
plot_accuracy_decrements(accuracy_results, 
                         overall_results,
                         labels=class_labels, 
                         save=True, 
                         x_label='Threshold Decrement Factor', 
                         title='CIFAR10 Testing Dataset Class Prediction Accuracy vs Distance to Threshold Decrement Factor', 
                         filename='CIFAR10_testing_plot_accuracy_decrements.png')


# Same as above CIFAR10 testing dataset, combined in a single plot
single_plot_accuracy_decrements(accuracy_results, 
                                overall_results, 
                                labels=class_labels, 
                                dataset="CIFAR10", 
                                save=True)

# STOPPED HERE
# Next steps:
# 1. Determine the accuracy of the model on the Training dataset given only examples from a threshold distance from the centroids
# 2. Determine the accuracy of the model on the Test dataset given only examples from a threshold distance from the centroids
# 3. Train 10 models on the training dataset, using softmax outputs as inputs, and the labels as targets, where given a digit, 
# the model predicts if the prediction is correct or not
# 4. Repeat step 3 for the test dataset
# 5. Based on the two metrics, distance to centroid and predictions, finish paper.
