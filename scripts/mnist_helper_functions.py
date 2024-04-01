import torch
import numpy as np

def save_predictions(net, testloader, verbose=False):
    """
    Save model predictions - softmax output, labels, and predicted labels - to a numpy array

    Parameters
    ----------
    net : torch.nn.Module
        The trained model.
    testloader : torch.utils.data.DataLoader
    verbose : bool, optional
    Output
    ------
    data_np : numpy.ndarray
        The softmax output, labels, and predicted labels for each image in the test dataset.
        Array shape: (number of images, 12)
    Example
    -------
    # Load the saved model from a file
    PATH = 'models/mnist_vanilla_cnn_local_202306241859.pth' # trained on google colab,
    PATH = os.path.join(current_dir, PATH)
    net.load_state_dict(torch.load(PATH))
    # Load the test data
    datapath = os.path.join(current_dir, 'data/')
    testset = datasets.MNIST(datapath, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    # Save the model predictions
    data_np = save_predictions(net, testloader)
    """

    # Running totals 
    correct = 0
    total = 0

    # Store Softmax output for each image prediction
    num_columns = 12
    data_np = np.empty((0, num_columns))

    # .no_grad() disables gradient computation, which reduces memory usage
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs_original = net(inputs)
            # convert log probabilities to probabilities
            outputs = torch.exp(outputs_original)
            _, predicted = torch.max(outputs.data, 1) # outputs.shape
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            outputs_np = outputs.data.numpy()
            labels_np = labels.numpy().reshape(-1, 1)
            predicted_np = predicted.numpy().reshape(-1, 1)
            combined_np = np.hstack((outputs_np, labels_np, predicted_np))
            data_np = np.vstack((data_np, combined_np))

    if verbose:
        # Print the accuracy of the model on the test dataset
        accuracy = 100 * correct / total # 98.38
        print('Accuracy on dataset: %.2f%%' % accuracy)

        # sanity check
        # Extracting the 11th and 12th columns (indexed as 10 and 11)
        labels = data_np[:, 10]
        predictions = data_np[:, 11]

        # Comparing the two columns to find where values are the same and where they are different
        same_values_mask = labels == predictions

        # Summing the values that are the same and different
        same_values_count = np.sum(same_values_mask)
        different_values_count = np.sum(~same_values_mask)

        # Calculating and printing the percentage of values that are the same
        percentage_same = (same_values_count / data_np.shape[0]) * 100
        print(f"Accuracy on dataset saved to numpy array: {percentage_same:.2f}%")

    return data_np

def find_unequal_rows(arr, idx1, idx2):
    
    # Compare the values at index 11 and 12
    unequal_mask = arr[:, idx1] != arr[:, idx2]

    # Get the row indices where the values are not equal
    unequal_indices = np.where(unequal_mask)[0]
    
    # Get the rows where the values are not equal
    unequal_rows = arr[unequal_mask]
    
    return unequal_indices, unequal_rows

###########
# DISPLAY #
###########

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def display_misclassifications(x, y_true, y_pred, save=False):
    """
    Displays a heatmap of the confusion matrix and example images of misclassified digits.

    Parameters:
    - x: Numpy array of MNIST images.
    - y_true: Numpy array containing the true labels.
    - y_pred: Numpy array containing the predicted labels.
    - save: Boolean, if True, saves the confusion matrix plot. Defaults to False.
    Example:
    display_misclassifications(x_test, y_test, y_pred, save=True)
    """
    # Retrieve data
    true_labels = x[:, y_true]
    predicted_labels = x[:, y_pred]    
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Create a mask for non-zero values
    non_zero_mask = np.array(conf_matrix > 0, dtype=int)

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    #sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', mask=non_zero_mask == 0)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', mask=non_zero_mask, cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    if save:
        plt.savefig('confusion_matrix.png')
    plt.show()

    import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def sum_misclassifications(conf_matrix):
    """
    Sums the misclassifications for each class in a confusion matrix.

    Parameters:
    - conf_matrix: A confusion matrix represented as a NumPy array.

    Returns:
    - A NumPy array containing the sum of misclassifications for each class.
    """
    # Create an array of the same shape as conf_matrix filled with the diagonal values
    diag = np.diag(conf_matrix)
    
    # Subtract the diagonal (correct classifications) from the sum of the entire row (total predictions for the class)
    misclassifications = np.sum(conf_matrix, axis=1) - diag
    
    return misclassifications

def display_misclassifications_side_by_side(x1, x2, y_true, y_pred, w=20, h=10, save=False):
    """
    Displays heatmaps of the confusion matrices for two datasets side by side and example images of misclassified digits.

    Parameters:
    - x1: Numpy array of MNIST images for the first dataset.
    - x2: Numpy array of MNIST images for the second dataset.
    - y_true: Numpy array containing the true labels for both datasets.
    - y_pred: Numpy array containing the predicted labels for both datasets.
    - w: Width of the figure. Defaults to 20.
    - h: Height of the figure. Defaults to 10.
    - save: Boolean, if True, saves the confusion matrix plot. Defaults to False.
    Returns:
    - The confusion matrices for both datasets.
    Example:
    conf_matrix1, confmatrix2 = display_misclassifications_side_by_side(x1_test, x2_test, y_test, y_pred, save=True)
    """
    # Retrieve data
    true_labels1 = x1[:, y_true]
    predicted_labels1 = x1[:, y_pred]   
    true_labels2 = x2[:, y_true]
    predicted_labels2 = x2[:, y_pred]      

    # Compute the confusion matrices for both datasets
    conf_matrix1 = confusion_matrix(true_labels1, predicted_labels1)
    conf_matrix2 = confusion_matrix(true_labels2, predicted_labels2)  # Assuming y_true, y_pred are applicable to both x1 and x2

    # Create a mask for non-zero values
    non_zero_mask1 = np.array(conf_matrix1 > 0, dtype=int)
    non_zero_mask2 = np.array(conf_matrix2 > 0, dtype=int)

    plt.figure(figsize=(w, h))

    # Plotting for the first dataset
    plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
    sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Reds', mask=non_zero_mask1 == 0)
    sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', mask=non_zero_mask1, cbar=False)
    plt.title('Confusion Matrix for MNIST Training Dataset')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Plotting for the second dataset
    plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
    sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Reds', mask=non_zero_mask2 == 0)
    sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', mask=non_zero_mask2, cbar=False)
    plt.title('Confusion Matrix for MNIST Testing Dataset')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Decrease the padding between the plots
    plt.subplots_adjust(wspace=0.1) 

    if save:
        plt.savefig('mnist_combined_confusion_matrix.png')
    
    plt.show()

    # For additional downstream data analysis
    return conf_matrix1, conf_matrix2

import numpy as np
import matplotlib.pyplot as plt

def plot_misclassifications(train_misclassifications, test_misclassifications, save=False):
    """
    Plots the normalised misclassifications for the training and testing datasets.
    Parameters:
    - train_misclassifications: Numpy array containing the sum of misclassifications for each class in the training dataset.
    - test_misclassifications: Numpy array containing the sum of misclassifications for each class in the testing dataset.
    - save: Boolean, if True, saves the plot. Defaults to False.
    Example:
    plot_misclassifications(train_misclassifications, test_misclassifications, save=True)
    """
    # Normalize the misclassification arrays
    train_normalized = train_misclassifications / np.sum(train_misclassifications)
    test_normalized = test_misclassifications / np.sum(test_misclassifications)

    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the training data misclassifications
    ax1.bar(range(10), train_normalized)
    ax1.set_title('Training Data Misclassifications')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Normalised Misclassifications')
    ax1.set_xticks(range(10))
    ax1.set_xticklabels([f'{i}\n{int(train_misclassifications[i])}' for i in range(10)])

    # Plot the testing data misclassifications
    ax2.bar(range(10), test_normalized)
    ax2.set_title('Testing Data Misclassifications')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Normalised Misclassifications')
    ax2.set_xticks(range(10))
    ax2.set_xticklabels([f'{i}\n{int(test_misclassifications[i])}' for i in range(10)])

    # Adjust the layout and display the plot
    plt.tight_layout()

    if save:
        plt.savefig('mnist_combined_misclassifications.png')
    
    plt.show()

from sklearn.cluster import KMeans

def get_kmeans_centroids(correct_preds, debug=False):
    """
    Get the KMeans centroids for the probability distributions of the correctly classified images.
    Parameters:
    - correct_preds: Numpy array containing the softmax output, labels, and predicted labels for each correctly classified image.
    - debug: Boolean, if True, prints the cluster labels and final centroids. Defaults to True.
    Returns:
    - final_centroids: Numpy array containing the final centroids after fitting the KMeans object.
    - cluster_labels: Numpy array containing the cluster labels for each data point.
    Example:
    centroids = get_kmeans_centroids(correct_preds, debug=True)
    """
    # Initialize an empty array for the centroids
    centroids = np.zeros((10, 10))  # 10 clusters for 10 digits, each centroid of size 10 (probability distribution)

    prob_dist = correct_preds[:, :10]

    for digit in range(10):
        # Extract rows where the highest probability corresponds to the digit class
        digit_rows = correct_preds[correct_preds[:, 10] == digit]
        
        # Calculate the mean of these rows to use as the centroid
        centroids[digit] = np.mean(digit_rows[:, :10], axis=0)

    # Create the KMeans object with 10 clusters and the specified centroids
    kmeans = KMeans(n_clusters=10, init=centroids, n_init=1)

    # Fit the KMeans object to the probability distribution data
    kmeans.fit(prob_dist)

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Get the final centroids after fitting the KMeans object
    final_centroids = kmeans.cluster_centers_

    if debug:
        # Print the cluster labels and final centroids
        print("Cluster Labels:")
        print(cluster_labels)
        print("\nFinal Centroids:")
        print(final_centroids)

    return final_centroids, cluster_labels

def find_mismatched_indexes(data, cluster_labels):
    """Finds the indexes of rows in a NumPy array where the value at column 11 matches column 12,
       and the corresponding element in the cluster labels differs.

    Args:
        data (numpy.ndarray): The input NumPy array.
        cluster_labels (numpy.ndarray): An array of cluster labels.

    Returns:
        numpy.ndarray: An array containing the indexes of the mismatched rows.

    Example:
    mismatched_index = find_mismatched_indexes(data_np, cluster_labels)
    """

    match_condition = data[:, 10] == data[:, 11]
    relevant_data = data[match_condition]

    mismatch_condition = relevant_data[:, 10] != cluster_labels 
    mismatched_index = np.where(mismatch_condition)[0]  

    return mismatched_index

from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def plot_clusters_with_centroids(train_np, centroids, cluster_labels):
    """
    Plots a 3D scatter plot of the clusters along with their centroids.
    
    Parameters:
    - train_np: Numpy array containing the dataset.
    - centroids: Numpy array containing the centroids of the clusters.
    - cluster_labels: Numpy array of cluster labels for each point in train_np.

    Example:
    plot_clusters_with_centroids(train_np, centroids, cluster_labels)
    """
    # Reduce the dimensionality of both the data and the centroids to three dimensions using PCA
    pca = PCA(n_components=3)

    match_condition = train_np[:, 10] == train_np[:, 11]
    relevant_data = train_np[match_condition]

    reduced_data = pca.fit_transform(relevant_data[:, :10])
    reduced_centroids = pca.transform(centroids[:, :10])

    # Plot the reduced data points with their cluster assignments in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for each cluster
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use tab10 colormap for 10 clusters
    for i in range(10):  # Assuming there are 10 clusters
        cluster_data = reduced_data[cluster_labels == i]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], 
                   label=f'Cluster {i}', c=[colors[i]], s=15)

    # Scatter plot for the centroids
    ax.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], reduced_centroids[:, 2], 
               s=100, c='black', label='Centroids', marker='x')

    ax.set_title('3D MNIST Probability Distribution Clusters with Centroids')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    ax.legend()
    plt.show()

def plot_2d_clusters_with_centroids(train_np, centroids, cluster_labels):

    # Assuming `train_np` and `cluster_labels` are defined
    # Assuming that train_np[:, :10] contains the features used for clustering

    # Reduce the dimensionality of the data to two dimensions using PCA
    pca = PCA(n_components=2)

    match_condition = train_np[:, 10] == train_np[:, 11]
    relevant_data = train_np[match_condition]
    reduced_data = pca.fit_transform(relevant_data[:, :10])
    reduced_centroids = pca.transform(centroids[:, :10])

    #reduced_data = pca.fit_transform(train_np[:, :10])

    # Plot the reduced data points with their cluster assignments
    plt.figure(figsize=(8, 6))
    for i in range(10):  # Assuming we have 10 clusters
        # Select only data points that belong to the current cluster
        cluster_data = reduced_data[cluster_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i}')

    plt.title('2D MNIST Probability Distribution Clusters')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.show()

# claude
def calculate_mean_distances(train_np, centroids):
    # Extract the probability distribution for each digit (first 10 columns of train_np)
    prob_dist = train_np[:, :10]
    
    # Get the labels from column 11 (index 10) of train_np
    labels = train_np[:, 10].astype(int)
    
    # Initialize an array to store the mean distances for each label
    mean_distances = np.zeros(10)
    
    # Iterate over each label (0 to 9)
    for label in range(10):
        # Find the indices of rows with the current label
        label_indices = np.where(labels == label)[0]
        
        # Calculate the Euclidean distances between the rows with the current label and the corresponding centroid
        distances = np.linalg.norm(prob_dist[label_indices] - centroids[label], axis=1)
        
        # Calculate the mean distance for the current label
        mean_distance = np.mean(distances)
        
        # Store the mean distance in the mean_distances array
        mean_distances[label] = mean_distance
    
    return mean_distances

# gpt
def mean_distance_to_centroids(train_np, centroids):
    """
    Calculates the mean distance of data points in each cluster to their corresponding centroid.

    Parameters:
    - train_np: Numpy array of training data, with the last column (index 10) as the label.
    - centroids: Numpy array of centroids.

    Returns:
    - A numpy array containing the mean distance of data points to their corresponding centroid for each cluster.
    """
    num_clusters = centroids.shape[0]
    distances = np.zeros(num_clusters)
    counts = np.zeros(num_clusters)

    # Iterate through each cluster
    for i in range(num_clusters):
        # Extract rows for the current cluster based on the label
        cluster_data = train_np[train_np[:, 10] == i][:, :10]  # Exclude the label column
        # Calculate the Euclidean distance from each point in the cluster to the centroid
        if len(cluster_data) > 0:
            dist = np.sqrt(np.sum((cluster_data - centroids[i])**2, axis=1))
            # Sum distances and count points for the current cluster
            distances[i] = np.mean(dist)
        else:
            distances[i] = np.nan  # Handle empty clusters if any

    return distances

import matplotlib.pyplot as plt

def plot_mean_distances(mean_distances):
    """
    Plots a bar chart of mean distances to centroids.

    Parameters:
    - mean_distances: A numpy array containing the mean distances to centroids for each cluster.
    """
    # Plot the mean distances directly without scaling
    plt.figure(figsize=(10, 6))
    clusters = range(len(mean_distances))  # Assuming mean_distances has a size of 10
    plt.bar(clusters, mean_distances, color='skyblue')
    
    # Adding labels and title
    plt.xlabel('Cluster (Digit)')
    plt.ylabel('Mean Distance to Centroid')
    plt.title('Mean Distance to Centroid for Each Cluster')
    plt.xticks(clusters, [str(i) for i in clusters])  # Label x-axis with cluster numbers
    
    # Annotate each bar with the mean distance value
    for i, distance in enumerate(mean_distances):
        plt.text(i, distance, f'{distance:.4f}', ha='center', va='bottom')
    
    plt.show()

def plot_mean_distances_x2(training_mean_distances, testing_mean_distances, predictions_type="Correct", save=False):
    """
    Plots bar charts of mean distances to centroids for training and testing datasets side by side.

    Parameters:
    - training_mean_distances: A numpy array containing the mean distances to centroids for each cluster in the training dataset.
    - testing_mean_distances: A numpy array containing the mean distances to centroids for each cluster in the testing dataset.
    - predictions_type: A string indicating the type of predictions (e.g., "Correct" or "Incorrect"). Default is "Correct".
    - save: Boolean, if True, saves the plot. Defaults to False.
    """
    # Plot the mean distances side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    clusters = range(len(training_mean_distances))  # Assuming both arrays have the same size

    # Plot training mean distances
    ax1.bar(clusters, training_mean_distances, color='skyblue')
    ax1.set_xlabel('Cluster (Digit)')
    ax1.set_ylabel('Mean Distance to Centroid')
    ax1.set_title(f'Training Data - {predictions_type} Predictions\nMean Distance to Centroid for Each Cluster')
    ax1.set_xticks(clusters)
    ax1.set_xticklabels([str(i) for i in clusters])

    # Annotate each bar with the mean distance value for training data
    for i, distance in enumerate(training_mean_distances):
        ax1.text(i, distance, f'{distance:.4f}', ha='center', va='bottom')

    # Plot testing mean distances
    ax2.bar(clusters, testing_mean_distances, color='lightgreen')
    ax2.set_xlabel('Cluster (Digit)')
    ax2.set_ylabel('Mean Distance to Centroid')
    ax2.set_title(f'Testing Data - {predictions_type} Predictions\nMean Distance to Centroid for Each Cluster')
    ax2.set_xticks(clusters)
    ax2.set_xticklabels([str(i) for i in clusters])

    # Annotate each bar with the mean distance value for testing data
    for i, distance in enumerate(testing_mean_distances):
        ax2.text(i, distance, f'{distance:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('Combined_mean_distances.png')

    import numpy as np

def calculate_distances_to_centroids(train_np, equal=True, debug=False):
    """
    Calculates the distances to centroids for correct or incorrect predictions in the training dataset.

    Parameters:
    - train_np: A numpy array containing the training data.
    - equal: A boolean indicating whether to consider equal or not equal predictions and labels.
             If True (default), considers correct predictions (labels == predictions).
             If False, considers incorrect predictions (labels != predictions).
    -debug: A boolean indicating whether to print debug information. Defaults to False.

    Returns:
    - distances_to_centroids: A numpy array containing the mean distances to centroids for each cluster, for each MNIST class.
    """
    if equal:
        # Consider correct predictions (labels == predictions)
        match_condition = train_np[:, 10] == train_np[:, 11]
    else:
        # Consider incorrect predictions (labels != predictions)
        match_condition = train_np[:, 10] != train_np[:, 11]

    # Filter the train_np array based on the match condition
    filtered_preds = train_np[match_condition]

    # Get the centroids and cluster labels using the get_kmeans_centroids function
    centroids, cluster_labels = get_kmeans_centroids(filtered_preds, debug=debug)

    # Calculate the distances to centroids using the mean_distance_to_centroids function
    distances_to_centroids = mean_distance_to_centroids(filtered_preds, centroids)

    return distances_to_centroids

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import pandas as pd

def plot_centroids(centroids):
  """
  Plots the centroids based on their dimensionality.

  Args:
      centroids (numpy.ndarray): An array containing the centroids.
  """
  num_centroids, num_features = centroids.shape

  if num_features == 2:
      # Scatter plot for 2D data
      plt.scatter(centroids[:, 0], centroids[:, 1])
      for i, centroid in enumerate(centroids):
          plt.text(centroid[0], centroid[1], str(i))
      plt.xlabel("Dimension 1")
      plt.ylabel("Dimension 2")
      plt.title("Visualization of Centroids in 2D")
      plt.show()
  elif num_features == 3:
      # 3D scatter plot
      fig = plt.figure(figsize=(8, 6))
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2])
      for i, centroid in enumerate(centroids):
          ax.text(centroid[0], centroid[1], centroid[2], str(i))
      ax.set_xlabel('X Label')
      ax.set_ylabel('Y Label')
      ax.set_zlabel('Z Label')
      plt.title("Visualization of Centroids in 3D")
      plt.show()
  elif num_features > 3:
      # Use parallel coordinates for higher dimensions
      from pandas.plotting import parallel_coordinates
      df = pd.DataFrame(centroids)
      parallel_coordinates(df, class_column=None, marker='o')
      plt.title("Parallel Coordinates Visualization of Centroids")
      plt.show()
  else:
      print("Centroids have invalid dimensionality for plotting. Choose a suitable visualization technique.")

def plot_mean_distances_double_bars(training_correct_distances, training_incorrect_distances,
                           testing_correct_distances, testing_incorrect_distances, save=False):
    """
    Plots bar charts of mean distances to centroids for training and testing datasets.
    Each class is represented by two bars side by side: correct and incorrect predictions.

    Parameters:
    - training_correct_distances: A numpy array containing the mean distances to centroids for correct predictions in the training dataset.
    - training_incorrect_distances: A numpy array containing the mean distances to centroids for incorrect predictions in the training dataset.
    - testing_correct_distances: A numpy array containing the mean distances to centroids for correct predictions in the testing dataset.
    - testing_incorrect_distances: A numpy array containing the mean distances to centroids for incorrect predictions in the testing dataset.
    - save: Boolean, if True, saves the plot. Defaults to False.
    """
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    clusters = range(len(training_correct_distances))

    # Plot training data
    bar_width = 0.35
    ax1.bar(np.arange(len(clusters)) - bar_width/2, training_correct_distances, bar_width, color='skyblue', label='Correctly classified')
    ax1.bar(np.arange(len(clusters)) + bar_width/2, training_incorrect_distances, bar_width, color='lightcoral', label='Incorrectly classified')
    ax1.set_xlabel('Digit Class Prediction')
    ax1.set_ylabel('Mean Distance to Centroid')
    ax1.set_title('Training Data - Softmax Output Mean Distance to Centroid')
    ax1.set_xticks(np.arange(len(clusters)))
    ax1.set_xticklabels([str(i) for i in clusters])
    ax1.legend(loc='center left') # ax1.legend()

    # Annotate each bar with the mean distance value for training data
    for i, (correct_distance, incorrect_distance) in enumerate(zip(training_correct_distances, training_incorrect_distances)):
        ax1.text(i - bar_width/2, correct_distance, f'{correct_distance:.4f}', ha='center', va='bottom')
        ax1.text(i + bar_width/2, incorrect_distance, f'{incorrect_distance:.4f}', ha='center', va='bottom')

    # Plot testing data
    ax2.bar(np.arange(len(clusters)) - bar_width/2, testing_correct_distances, bar_width, color='lightgreen', label='Correctly classified')
    ax2.bar(np.arange(len(clusters)) + bar_width/2, testing_incorrect_distances, bar_width, color='lightcoral', label='Incorrectly classified')
    ax2.set_xlabel('Digit Class Prediction')
    ax2.set_ylabel('Mean Distance to Centroid')
    ax2.set_title('Testing Data - Softmax Output Mean Distance to Centroid')
    ax2.set_xticks(np.arange(len(clusters)))
    ax2.set_xticklabels([str(i) for i in clusters])
    ax2.legend(loc='center left') # ax2.legend()

    # Annotate each bar with the mean distance value for testing data
    for i, (correct_distance, incorrect_distance) in enumerate(zip(testing_correct_distances, testing_incorrect_distances)):
        ax2.text(i - bar_width/2, correct_distance, f'{correct_distance:.4f}', ha='center', va='bottom')
        ax2.text(i + bar_width/2, incorrect_distance, f'{incorrect_distance:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig('Combined_mean_distances_double_bars.png')


def mean_distance_to_centroids_boxplots(train_np, centroids, plot=False):
    """
    Calculates the mean distance of data points in each cluster to their corresponding centroid.
    
    Parameters:
    - train_np: Numpy array of training data, with the last column (index 10) as the label.
    - centroids: Numpy array of centroids.
    - plot: Boolean, if True, plots a box plot of the distances for each class. Default is False.
    
    Returns:
    - A numpy array containing the mean distance of data points to their corresponding centroid for each cluster.
    """
    num_clusters = centroids.shape[0]
    distances = np.zeros(num_clusters)
    counts = np.zeros(num_clusters)
    
    # Create a list to store distances for each class
    class_distances = [[] for _ in range(num_clusters)]
    
    # Iterate through each cluster
    for i in range(num_clusters):
        # Extract rows for the current cluster based on the label
        cluster_data = train_np[train_np[:, 10] == i][:, :10]  # Exclude the label column
        
        # Calculate the Euclidean distance from each point in the cluster to the centroid
        if len(cluster_data) > 0:
            dist = np.sqrt(np.sum((cluster_data - centroids[i])**2, axis=1))
            # Sum distances and count points for the current cluster
            distances[i] = np.mean(dist)
            # Append distances to the corresponding class list
            class_distances[i].extend(dist)
        else:
            distances[i] = np.nan  # Handle empty clusters if any
    
    if plot:
        # Create a box plot of distances for each class
        plt.figure(figsize=(10, 6))
        plt.boxplot(class_distances, labels=[str(i) for i in range(num_clusters)])
        plt.xlabel('Class')
        plt.ylabel('Distance to Centroid')
        plt.title('Distribution of Distances to Centroids')
        plt.show()
    
    return distances        

def calculate_class_accuracies(data):
    """
    Calculate the accuracy for each class (digit 0 to 9) based on the comparison of true labels and predictions.

    Parameters:
    - data: Numpy array with 12 columns, where column 11 (index 10) contains the true labels
            and column 12 (index 11) contains the predicted labels.

    Returns:
    - accuracies: Numpy array with one row and 10 columns, where each column represents a class (digit 0 to 9)
                  and the value is the accuracy for that class.
    """
    num_classes = 10
    accuracies = np.zeros((1, num_classes))

    for class_label in range(num_classes):
        # Get the rows where the true label (column 11) matches the current class label
        class_mask = data[:, 10] == class_label
        class_rows = data[class_mask]

        if len(class_rows) > 0:
            # Count the number of rows where the true label (column 11) matches the prediction (column 12)
            correct_predictions = np.sum(class_rows[:, 10] == class_rows[:, 11])
            
            # Calculate the accuracy for the current class
            accuracy = correct_predictions / len(class_rows)
            accuracies[0, class_label] = accuracy

    return accuracies

import numpy as np

def calculate_class_accuracies_2(data):
    """
    Calculates the accuracy for each class based on the comparison between true labels and predictions.

    Parameters:
    - data: A numpy array with 12 columns, where the last two columns are the true label and the prediction, respectively.

    Returns:
    - A numpy array of size 10, containing the accuracy for each class from 0 to 9.
    """
    # Initialize an array to hold the accuracy for each class
    accuracies = np.zeros(10)
    
    # Iterate through each class (digit)
    for i in range(10):
        # Select rows for the current class
        class_rows = data[data[:, 10] == i]
        # Count the correct predictions for the current class
        correct_predictions = np.sum(class_rows[:, 10] == class_rows[:, 11])
        # Calculate accuracy as the number of correct predictions divided by the number of rows for the class
        if len(class_rows) > 0:
            accuracies[i] = correct_predictions / len(class_rows)
        else:
            accuracies[i] = np.nan  # Avoid division by zero if no rows for the current class

    return accuracies

# Example usage:
# Assume 'data' is a numpy array with 12 columns, where column 10 is the true class and column 11 is the prediction
# data = np.array([...])  # Replace with actual data
# class_accuracies = calculate_class_accuracies(data)

import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_vs_distance(distance, accuracy):
    """
    Plot the accuracy on the Y-axis and the distance on the X-axis.

    Parameters:
    - distance: Numpy array of shape (10,) representing the mean class distance to the centroid.
    - accuracy: Numpy array of shape (1, 10) representing the accuracy for each class.

    Returns:
    - None
    """
    # Reshape the arrays if necessary
    distance = distance.reshape(-1)
    accuracy = accuracy.reshape(-1)

    # Calculate the minimum and maximum values for each array
    min_distance = np.min(distance)
    max_distance = np.max(distance)
    min_accuracy = np.min(accuracy)
    max_accuracy = np.max(accuracy)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the accuracy vs. distance
    ax.plot(distance, accuracy, marker='o', linestyle='-', linewidth=2, markersize=8)

    # Set the X-axis label and scale
    ax.set_xlabel("Mean Class Distance to Centroid")
    ax.set_xlim(min_distance * 0.9, max_distance * 1.1)  # Adjust the limits for better visibility
    ax.set_xscale('log')  # Use logarithmic scaling for the X-axis

    # Set the Y-axis label and scale
    ax.set_ylabel("Accuracy")
    ax.set_ylim(min_accuracy * 0.9, max_accuracy * 1.1)  # Adjust the limits for better visibility

    # Set the title
    ax.set_title("MNIST Classification Softmax Output, Accuracy vs. Mean Distance to Centroid")

    # Display the plot
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_vs_distance_combined(train_distance, train_accuracy, test_distance, test_accuracy):
    """
    Plot the accuracy on the Y-axis and the distance on the X-axis as dots without connecting them,
    for both training and testing data.

    Parameters:
    - train_distance: Numpy array of shape (10,) representing the mean class distance to the centroid for training data.
    - train_accuracy: Numpy array of shape (1, 10) representing the accuracy for each class for training data.
    - test_distance: Numpy array of shape (10,) representing the mean class distance to the centroid for testing data.
    - test_accuracy: Numpy array of shape (1, 10) representing the accuracy for each class for testing data.

    Returns:
    - None
    """
    # Reshape the arrays if necessary
    train_distance = train_distance.reshape(-1)
    train_accuracy = train_accuracy.reshape(-1)
    test_distance = test_distance.reshape(-1)
    test_accuracy = test_accuracy.reshape(-1)

    # Calculate the minimum and maximum values for each array
    min_distance = np.min(np.concatenate((train_distance, test_distance)))
    max_distance = np.max(np.concatenate((train_distance, test_distance)))

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the training accuracy vs. distance as blue dots without connecting them
    ax.plot(train_distance, train_accuracy, marker='o', linestyle='', markersize=8, color='blue', label='Training')

    # Plot the testing accuracy vs. distance as red dots without connecting them
    ax.plot(test_distance, test_accuracy, marker='o', linestyle='', markersize=8, color='red', label='Testing')

    # Set the X-axis label and scale
    ax.set_xlabel("Mean Class Distance to Centroid")
    ax.set_xlim(min_distance * 0.9, max_distance * 1.1)  # Adjust the limits for better visibility
    ax.set_xscale('log')  # Use logarithmic scaling for the X-axis

    # Set the Y-axis label and limits
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.94, 1.0)  # Set the Y-axis limits between 0.94 and 1

    # Set the title
    ax.set_title("MNIST Classification Softmax Output, Accuracy vs. Mean Distance to Centroid")

    # Add a grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # Add a legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()    

def linear_func(x, a, b):
    return a * x + b

def plot_accuracy_vs_distance_linear(train_distance, train_accuracy, test_distance, test_accuracy):
    """
    Plot the accuracy on the Y-axis and the distance on the X-axis as dots without connecting them,
    for both training and testing data. Fit a linear function to the data and plot the functions.

    Parameters:
    - train_distance: Numpy array of shape (10,) representing the mean class distance to the centroid for training data.
    - train_accuracy: Numpy array of shape (1, 10) representing the accuracy for each class for training data.
    - test_distance: Numpy array of shape (10,) representing the mean class distance to the centroid for testing data.
    - test_accuracy: Numpy array of shape (1, 10) representing the accuracy for each class for testing data.

    Returns:
    - None
    """
    # Reshape the arrays if necessary
    train_distance = train_distance.reshape(-1)
    train_accuracy = train_accuracy.reshape(-1)
    test_distance = test_distance.reshape(-1)
    test_accuracy = test_accuracy.reshape(-1)

    # Calculate the minimum and maximum values for each array
    min_distance = np.min(np.concatenate((train_distance, test_distance)))
    max_distance = np.max(np.concatenate((train_distance, test_distance)))

    # Fit a linear function to the training data
    train_popt, _ = np.polyfit(train_distance, train_accuracy, 1, full=True)
    train_func = lambda x: linear_func(x, train_popt[0], train_popt[1])

    # Fit a linear function to the testing data
    test_popt, _ = np.polyfit(test_distance, test_accuracy, 1, full=True)
    test_func = lambda x: linear_func(x, test_popt[0], test_popt[1])

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the training accuracy vs. distance as blue dots without connecting them
    ax.plot(train_distance, train_accuracy, marker='o', linestyle='', markersize=8, color='blue', label='Training')

    # Plot the testing accuracy vs. distance as red dots without connecting them
    ax.plot(test_distance, test_accuracy, marker='o', linestyle='', markersize=8, color='red', label='Testing')

    # Plot the fitted functions
    x_vals = np.linspace(min_distance, max_distance, 100)
    ax.plot(x_vals, train_func(x_vals), color='blue', linestyle='-', label='Training Fit')
    ax.plot(x_vals, test_func(x_vals), color='red', linestyle='-', label='Testing Fit')

    # Set the X-axis label and scale
    ax.set_xlabel("Mean Class Distance to Centroid")
    ax.set_xlim(min_distance * 0.9, max_distance * 1.1)  # Adjust the limits for better visibility
    ax.set_xscale('log')  # Use logarithmic scaling for the X-axis

    # Set the Y-axis label and limits
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.94, 1.0)  # Set the Y-axis limits between 0.94 and 1

    # Set the title
    ax.set_title("MNIST Classification Softmax Output, Accuracy vs. Mean Distance to Centroid")

    # Add a grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # Add a legend
    ax.legend()

    # Add the function equations inside the plot on the bottom left
    train_eq = f"$acc_{{train}} = {train_popt[0]:.3e} * dist + {train_popt[1]:.3f}$"
    test_eq = f"$acc_{{test}} = {test_popt[0]:.3e} * dist + {test_popt[1]:.3f}$"
    ax.text(0.02, 0.02, train_eq + '\n' + test_eq, transform=ax.transAxes, fontsize=12, verticalalignment='bottom')

    # Display the plot
    plt.tight_layout()
    plt.show()

from scipy.optimize import curve_fit

def exponential_fit(x, a, b, c):
    return a * np.exp(b * x) + c

def plot_accuracy_vs_distance_fitted(train_distance, train_accuracy, test_distance, test_accuracy):
    """
    Plots accuracy against mean class distance to centroid for both training and testing data,
    with an exponential fit for the trend in the data.

    Parameters:
    - train_distance: A numpy array with shape (10,) containing mean distances to centroid for each class in the training set.
    - train_accuracy: A numpy array with shape (1,10) containing accuracies for each class in the training set.
    - test_distance: A numpy array with shape (10,) containing mean distances to centroid for each class in the testing set.
    - test_accuracy: A numpy array with shape (1,10) containing accuracies for each class in the testing set.
    """
    # Ensure the arrays are correctly shaped
    train_distance = np.reshape(train_distance, (10,))
    train_accuracy = np.reshape(train_accuracy, (10,))
    test_distance = np.reshape(test_distance, (10,))
    test_accuracy = np.reshape(test_accuracy, (10,))

    # Fit an exponential decay function to the data
    params_train, _ = curve_fit(exponential_fit, train_distance, train_accuracy, maxfev=10000)
    params_test, _ = curve_fit(exponential_fit, test_distance, test_accuracy, maxfev=10000)

    # Generate a sequence of distances for plotting the fit function
    distance_plot = np.linspace(min(train_distance.min(), test_distance.min()), max(train_distance.max(), test_distance.max()), 100)

    # Calculate the fitted values
    fit_train = exponential_fit(distance_plot, *params_train)
    fit_test = exponential_fit(distance_plot, *params_test)

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot train data and fit function
    plt.scatter(train_distance, train_accuracy, c='blue', label='Train Data')
    plt.plot(distance_plot, fit_train, 'b--', label='Train Fit: {:.3f}*exp({:.3f}*x)+{:.3f}'.format(*params_train))

    # Plot test data and fit function
    plt.scatter(test_distance, test_accuracy, c='green', label='Test Data')
    plt.plot(distance_plot, fit_test, 'g--', label='Test Fit: {:.3f}*exp({:.3f}*x)+{:.3f}'.format(*params_test))

    # Set the axis labels
    plt.xlabel('Mean Class Distance to Centroid')
    plt.ylabel('Accuracy')

    # Set the y-axis limits between 0.94 and 1
    plt.ylim(0.94, 1)

    # Set the title
    plt.title('MNIST Classification: Train vs Test Accuracy and Mean Distance to Centroid with Exponential Fit')

    # Show the grid
    plt.grid(True)

    # Show the legend with fit functions
    plt.legend(loc='lower left')

    # Show the plot
    plt.show()

# Example usage with dummy data
# train_distance = np.array([...])
# train_accuracy = np.array([...])
# test_distance = np.array([...])
# test_accuracy = np.array([...])
# plot_accuracy_vs_distance_fitted(train_distance, train_accuracy, test_distance, test_accuracy)
    
def linear_fit(x, m, c):
    return m * x + c

def plot_accuracy_vs_distance_linear_fit(train_distance, train_accuracy, test_distance, test_accuracy):
    """
    Plots accuracy against mean class distance to centroid for both training and testing data,
    with a linear fit for the trend in the data.

    Parameters:
    - train_distance: A numpy array with shape (10,) containing mean distances to centroid for each class in the training set.
    - train_accuracy: A numpy array with shape (1,10) containing accuracies for each class in the training set.
    - test_distance: A numpy array with shape (10,) containing mean distances to centroid for each class in the testing set.
    - test_accuracy: A numpy array with shape (1,10) containing accuracies for each class in the testing set.
    """
    # Ensure the arrays are correctly shaped
    train_distance = np.reshape(train_distance, (10,))
    train_accuracy = np.reshape(train_accuracy, (10,))
    test_distance = np.reshape(test_distance, (10,))
    test_accuracy = np.reshape(test_accuracy, (10,))

    # Fit a linear function to the data
    params_train, _ = curve_fit(linear_fit, train_distance, train_accuracy)
    params_test, _ = curve_fit(linear_fit, test_distance, test_accuracy)

    # Generate a sequence of distances for plotting the fit function
    distance_plot = np.linspace(min(train_distance.min(), test_distance.min()), max(train_distance.max(), test_distance.max()), 100)

    # Calculate the fitted values
    fit_train = linear_fit(distance_plot, *params_train)
    fit_test = linear_fit(distance_plot, *params_test)

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot train data and fit function
    plt.scatter(train_distance, train_accuracy, c='blue', label='Train Data')
    plt.plot(distance_plot, fit_train, 'b--', label=f'Train Fit: {params_train[0]:.3f}*x + {params_train[1]:.3f}')

    # Plot test data and fit function
    plt.scatter(test_distance, test_accuracy, c='green', label='Test Data')
    plt.plot(distance_plot, fit_test, 'g--', label=f'Test Fit: {params_test[0]:.3f}*x + {params_test[1]:.3f}')

    # Set the axis labels
    plt.xlabel('Mean Class Distance to Centroid')
    plt.ylabel('Accuracy')

    # Set the y-axis limits between 0.94 and 1
    plt.ylim(0.95, 1)

    # Set the title
    plt.title('MNIST Classification: Train vs Test Accuracy and Mean Distance to Centroid with Linear Fit')

    # Show the grid
    plt.grid(True)

    # Show the legend with fit functions
    plt.legend(loc='lower left')

    # Show the plot
    plt.show()

import numpy as np

def calculate_distances_to_centroids(train_correct_predictions, centroids):
    """
    Calculate the distances between the softmax outputs and their corresponding centroids.

    Args:
        train_correct_predictions (numpy.ndarray): Array of shape (59074, 12) containing softmax outputs
            and correct predictions for MNIST training images. Indexes 0 to 9 are the predictions,
            and index 10 is the correct prediction.
        centroids (numpy.ndarray): Array of shape (10, 10) containing cluster centroids. Row indexes
            correspond to the digit represented by the centroid.

    Returns:
        numpy.ndarray: Array of shape (59074, 2) where the first column contains the calculated distances,
            and the second column contains the correct predictions.
    """
    results = np.zeros((train_correct_predictions.shape[0], 2))

    for i, row in enumerate(train_correct_predictions):
        # Get the correct prediction (label) for this row
        correct_prediction = int(row[10])

        # Calculate the distance between the softmax outputs (row[:10]) and the corresponding centroid
        distance = np.linalg.norm(row[:10] - centroids[correct_prediction])

        # Store the distance and correct prediction in the results array
        results[i, 0] = distance
        results[i, 1] = correct_prediction

    return results

def plot_centroid_mean_distance_boxplots(data, save=False):
    # Extract unique labels from the second column
    labels = np.unique(data[:, 1])

    # Create a list to store the distances for each label
    distances_by_label = [data[data[:, 1] == label, 0] for label in labels]

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the box plots
    ax.boxplot(distances_by_label, labels=labels)

    # Set the title and labels
    ax.set_title('Distribution of Distances to Centroids for MNIST Digits')
    ax.set_xlabel('Digit')
    ax.set_ylabel('Distance')

    # Adjust the spacing between box plots
    plt.tight_layout()

    # Save the plot if save is set to True
    if save:
        plt.savefig('centroid_distances.png')

    # Display the plot
    plt.show()

def plot_centroid_mean_distance_boxplots_2(data, save=False, debug=False):
    # Extract unique labels from the second column
    labels = np.unique(data[:, 1]).astype(int) 

    # Create a list to store the distances for each label
    distances_by_label = [data[data[:, 1] == label, 0] for label in labels]

    if debug:
        analyze_lists_by_label(distances_by_label)
    
    # Boxplot Creation with Styling
    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(distances_by_label, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True, medianprops={'linewidth': 2},
                    showfliers=False) 

    # Customize Colors (simpler than cycling through list)
    colors = ['lightblue', 'lightgreen', 'pink', 'peachpuff', 'lavender', 
              'salmon', 'skyblue', 'khaki', 'mediumpurple', 'hotpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Tweak Line Styles
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='gray', linewidth=1.5) 

    # Customize Outlier ('flier') Style 
    plt.setp(bp['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)

    # Customize Mean Marker (diamond)
    plt.setp(bp['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)

    # Labels, Title, and Layout (Modified)
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    ax.set_title('Distribution of Distances to Centroids for MNIST Digits', fontsize=14)
    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Distance (Logarithmic Scale)', fontsize=12)  # Update y label
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    # Saving and Displaying
    if save:
        plt.savefig('centroid_distances_styled.png', dpi=300)
    plt.show()

def plot_side_by_side_boxplots_gemini(data1, data2, save=False, debug=False):
    """
    Plots boxplots for two datasets side by side for comparison.

    Args:
        data1 (numpy.ndarray): Array of shape (N, 2) containing distances and labels for the first dataset.
        data2 (numpy.ndarray): Array of shape (N, 2) containing distances and labels for the second dataset.
        save (bool, optional): If True, saves the plot. Defaults to False.
        debug (bool, optional): If True, prints descriptive statistics. Defaults to False.
    """

    def extract_data(data):
        labels = np.unique(data[:, 1]).astype(int)
        distances_by_label = [data[data[:, 1] == label, 0] for label in labels]
        return labels, distances_by_label

    # Extract data for both datasets
    labels1, distances_by_label1 = extract_data(data1)
    labels2, distances_by_label2 = extract_data(data2)

    if debug:
        analyze_lists_by_label(distances_by_label1)
        analyze_lists_by_label(distances_by_label2)

    # Creating two side-by-side boxplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Boxplot for data1
    ax1.boxplot(distances_by_label1, labels=labels1, patch_artist=True,
                showmeans=True, meanline=True, medianprops={'linewidth': 2}, showfliers=False)
    # Customize colors & styles (see previous explanations) ... 

    # Boxplot for data2
    ax2.boxplot(distances_by_label2, labels=labels2, patch_artist=True,
                showmeans=True, meanline=True, medianprops={'linewidth': 2}, showfliers=False)
    # Customize colors & styles (see previous explanations) ... 

    # Common settings and labels
    fig.suptitle('Comparison of Distance Distributions to Centroids', fontsize=14)
    for ax in (ax1, ax2):
        ax.set_yscale('log') 
        ax.set_xlabel('Digit', fontsize=12)
        ax.set_ylabel('Distance (Logarithmic Scale)', fontsize=12) 
        ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout() 

    # Saving and displaying
    if save:
        plt.savefig('centroid_distances_comparison.png', dpi=300)
    plt.show()

def plot_side_by_side_claude(data1, data2, save=False, debug=False):
    # Extract unique labels from the second column of data1 and data2
    labels1 = np.unique(data1[:, 1]).astype(int)
    labels2 = np.unique(data2[:, 1]).astype(int)

    # Create lists to store the distances for each label in data1 and data2
    distances_by_label1 = [data1[data1[:, 1] == label, 0] for label in labels1]
    distances_by_label2 = [data2[data2[:, 1] == label, 0] for label in labels2]

    if debug:
        analyze_lists_by_label(distances_by_label1)
        analyze_lists_by_label(distances_by_label2)

    # Boxplot Creation with Styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot boxplots for data1
    bp1 = ax1.boxplot(distances_by_label1, labels=labels1, patch_artist=True,
                      showmeans=True, meanline=True, medianprops={'linewidth': 2},
                      showfliers=False)

    # Plot boxplots for data2
    bp2 = ax2.boxplot(distances_by_label2, labels=labels2, patch_artist=True,
                      showmeans=True, meanline=True, medianprops={'linewidth': 2},
                      showfliers=False)

    # Customize Colors (simpler than cycling through list)
    colors = ['lightblue', 'lightgreen', 'pink', 'peachpuff', 'lavender',
              'salmon', 'skyblue', 'khaki', 'mediumpurple', 'hotpink']

    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)

    # Tweak Line Styles
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp1[element], color='gray', linewidth=1.5)
        plt.setp(bp2[element], color='gray', linewidth=1.5)

    # Customize Outlier ('flier') Style
    plt.setp(bp1['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
    plt.setp(bp2['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)

    # Customize Mean Marker (diamond)
    plt.setp(bp1['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)
    plt.setp(bp2['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)

    # Labels, Title, and Layout (Modified)
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_yscale('log')  # Set y-axis to logarithmic scale

    ax1.set_title('Data 1: Distribution of Distances to Centroids', fontsize=14)
    ax2.set_title('Data 2: Distribution of Distances to Centroids', fontsize=14)

    ax1.set_xlabel('Digit', fontsize=12)
    ax2.set_xlabel('Digit', fontsize=12)

    ax1.set_ylabel('Distance (Logarithmic Scale)', fontsize=12)  # Update y label
    ax2.set_ylabel('Distance (Logarithmic Scale)', fontsize=12)  # Update y label

    ax1.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)

    plt.tight_layout()

    # Saving and Displaying
    if save:
        plt.savefig('centroid_distances_styled_comparison.png', dpi=300)

    plt.show()

def boxplots_side_by_side(data1, data2, save=False, debug=False, **kwargs):
    # Extract unique labels from the second column of data1 and data2
    labels1 = np.unique(data1[:, 1]).astype(int)
    labels2 = np.unique(data2[:, 1]).astype(int)

    # Create lists to store the distances for each label in data1 and data2
    distances_by_label1 = [data1[data1[:, 1] == label, 0] for label in labels1]
    distances_by_label2 = [data2[data2[:, 1] == label, 0] for label in labels2]

    if debug:
        analyze_lists_by_label(distances_by_label1)
        analyze_lists_by_label(distances_by_label2)

    # Boxplot Creation with Styling
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot boxplots for data1
    bp1 = ax.boxplot(distances_by_label1, positions=np.arange(len(labels1))-0.2, widths=0.4,
                     patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2},
                     showfliers=False)

    # Plot boxplots for data2
    bp2 = ax.boxplot(distances_by_label2, positions=np.arange(len(labels2))+0.2, widths=0.4,
                     patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2},
                     showfliers=False)

    # Set colors for data1 and data2
    color1 = 'skyblue'
    color2 = 'lightcoral'

    for patch in bp1['boxes']:
        patch.set_facecolor(color1)

    for patch in bp2['boxes']:
        patch.set_facecolor(color2)

    # Tweak Line Styles
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp1[element], color='gray', linewidth=1.5)
        plt.setp(bp2[element], color='gray', linewidth=1.5)

    # Customize Outlier ('flier') Style
    plt.setp(bp1['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
    plt.setp(bp2['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)

    # Customize Mean Marker (diamond)
    plt.setp(bp1['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)
    plt.setp(bp2['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)

    # Labels, Title, and Layout (Modified)
    ax.set_yscale('log')  # Set y-axis to logarithmic scale

    # Set y-axis limits and tick positions
    ylim_min = min(ax.get_ylim()[0], ax.get_ylim()[0])
    ylim_max = max(ax.get_ylim()[1], ax.get_ylim()[1])
    yticks = [1e-2, 1e-1, 1e0]
    yticklabels = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$']

    ax.set_ylim(ylim_min, ylim_max)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.set_title(kwargs.get('title', 'Distribution of Distances to Centroids'), fontsize=14)
    ax.set_xlabel(kwargs.get('xlabel', 'Digit'), fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Distance (Logarithmic Scale)'), fontsize=12)
    ax.set_xticks(range(len(labels1)))
    ax.set_xticklabels(labels1)

    ax.tick_params(axis='both', labelsize=10)

    # Create legend
    legend_labels = ['Data 1', 'Data 2']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color1), plt.Rectangle((0, 0), 1, 1, facecolor=color2)]
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)

    plt.tight_layout()

    # Saving and Displaying
    if save:
        filename = kwargs.get('filename', 'centroid_distances_comparison.png')
        plt.savefig(filename, dpi=300)

    plt.show()

def boxplots_side_by_side_x2(data1, data2, data3, data4, save=False, debug=False, **kwargs):
    # Extract unique labels from the second column of data1, data2, data3, and data4
    labels1 = np.unique(data1[:, 1]).astype(int)
    labels2 = np.unique(data2[:, 1]).astype(int)
    labels3 = np.unique(data3[:, 1]).astype(int)
    labels4 = np.unique(data4[:, 1]).astype(int)

    # Create lists to store the distances for each label in data1, data2, data3, and data4
    distances_by_label1 = [data1[data1[:, 1] == label, 0] for label in labels1]
    distances_by_label2 = [data2[data2[:, 1] == label, 0] for label in labels2]
    distances_by_label3 = [data3[data3[:, 1] == label, 0] for label in labels3]
    distances_by_label4 = [data4[data4[:, 1] == label, 0] for label in labels4]

    if debug:
        analyze_lists_by_label(distances_by_label1)
        analyze_lists_by_label(distances_by_label2)
        analyze_lists_by_label(distances_by_label3)
        analyze_lists_by_label(distances_by_label4)

    # Boxplot Creation with Styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # Plot boxplots for data1 and data2 on the first subplot
    bp1 = ax1.boxplot(distances_by_label1, positions=np.arange(len(labels1))-0.2, widths=0.4,
                      patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2})
    bp2 = ax1.boxplot(distances_by_label2, positions=np.arange(len(labels2))+0.2, widths=0.4,
                      patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2})

    # Plot boxplots for data3 and data4 on the second subplot
    bp3 = ax2.boxplot(distances_by_label3, positions=np.arange(len(labels3))-0.2, widths=0.4,
                      patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2})
    bp4 = ax2.boxplot(distances_by_label4, positions=np.arange(len(labels4))+0.2, widths=0.4,
                      patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2})


    # Set colors for data1, data2, data3, and data4
    color1 = 'skyblue'
    color2 = 'lightcoral'
    color3 = 'lightgreen'
    color4 = 'lightcoral'

    for patch in bp1['boxes']:
        patch.set_facecolor(color1)
    for patch in bp2['boxes']:
        patch.set_facecolor(color2)
    for patch in bp3['boxes']:
        patch.set_facecolor(color3)
    for patch in bp4['boxes']:
        patch.set_facecolor(color4)

    # Tweak Line Styles
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp1[element], color='gray', linewidth=1.5)
        plt.setp(bp2[element], color='gray', linewidth=1.5)
        plt.setp(bp3[element], color='gray', linewidth=1.5)
        plt.setp(bp4[element], color='gray', linewidth=1.5)

    # Customize Outlier ('flier') Style
    plt.setp(bp1['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
    plt.setp(bp2['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
    plt.setp(bp3['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
    plt.setp(bp4['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)

    # Customize Mean Marker (diamond)
    plt.setp(bp1['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)
    plt.setp(bp2['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)
    plt.setp(bp3['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)
    plt.setp(bp4['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)

    # Labels, Title, and Layout (Modified)
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax2.set_yscale('log')

    # Set y-axis limits and tick positions for the first subplot
    ylim_min1 = min(ax1.get_ylim()[0], ax1.get_ylim()[0])
    ylim_max1 = max(ax1.get_ylim()[1], ax1.get_ylim()[1])
    yticks1 = [1e-2, 1e-1, 1e0]
    yticklabels1 = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$']
    ax1.set_ylim(ylim_min1, ylim_max1)
    ax1.set_yticks(yticks1)
    ax1.set_yticklabels(yticklabels1)

    # Set y-axis limits and tick positions for the second subplot
    ylim_min2 = min(ax2.get_ylim()[0], ax2.get_ylim()[0])
    ylim_max2 = max(ax2.get_ylim()[1], ax2.get_ylim()[1])
    yticks2 = [1e-2, 1e-1, 1e0]
    yticklabels2 = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$']
    ax2.set_ylim(ylim_min2, ylim_max2)
    ax2.set_yticks(yticks2)
    ax2.set_yticklabels(yticklabels2)

    ax1.set_title(kwargs.get('title1', ' Training Data Distribution of Distances to Centroids'), fontsize=14)
    ax2.set_title(kwargs.get('title2', 'Testing Data Distribution of Distances to Centroids'), fontsize=14)
    ax1.set_xlabel(kwargs.get('xlabel', 'Digit Class Prediction'), fontsize=12)
    ax2.set_xlabel(kwargs.get('xlabel', 'Digit Class Prediction'), fontsize=12)
    ax1.set_ylabel(kwargs.get('ylabel', 'Distance (Logarithmic Scale)'), fontsize=12)
    ax2.set_ylabel(kwargs.get('ylabel', 'Distance (Logarithmic Scale)'), fontsize=12)
    ax1.set_xticks(range(len(labels1)))
    ax1.set_xticklabels(labels1)
    ax2.set_xticks(range(len(labels3)))
    ax2.set_xticklabels(labels3)

    ax1.tick_params(axis='both', labelsize=10)
    ax2.tick_params(axis='both', labelsize=10)

    # Create legends for the first subplot
    legend_labels1 = ['Correctly classified', 'Incorrectly classified']
    legend_handles1 = [plt.Rectangle((0, 0), 1, 1, facecolor=color1), plt.Rectangle((0, 0), 1, 1, facecolor=color2)]
    ax1.legend(legend_handles1, legend_labels1, loc='lower right', fontsize=12)

    # Create legends for the second subplot
    legend_labels2 = ['Correctly classified', 'Incorrectly classified']
    legend_handles2 = [plt.Rectangle((0, 0), 1, 1, facecolor=color3), plt.Rectangle((0, 0), 1, 1, facecolor=color4)]
    ax2.legend(legend_handles2, legend_labels2, loc='lower right', fontsize=12)

    plt.subplots_adjust(wspace=0.2)  # Adjust spacing between subplots

    # Saving and Displaying
    if save:
        filename = kwargs.get('filename', 'centroid_distances_comparison_4datasets.png')
        plt.savefig(filename, dpi=300)

    plt.show()    

def plot_centroid_mean_distance_boxplots_2(data1, data2, save=False, debug=False, **kwargs):
    # Extract unique labels from the second column of data1 and data2
    labels1 = np.unique(data1[:, 1]).astype(int)
    labels2 = np.unique(data2[:, 1]).astype(int)

    # Create lists to store the distances for each label in data1 and data2
    distances_by_label1 = [data1[data1[:, 1] == label, 0] for label in labels1]
    distances_by_label2 = [data2[data2[:, 1] == label, 0] for label in labels2]

    if debug:
        analyze_lists_by_label(distances_by_label1)
        analyze_lists_by_label(distances_by_label2)

    # Boxplot Creation with Styling
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot boxplots for data1
    bp1 = ax.boxplot(distances_by_label1, positions=np.arange(len(labels1))-0.2, widths=0.4,
                     patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2},
                     showfliers=False)

    # Plot boxplots for data2
    bp2 = ax.boxplot(distances_by_label2, positions=np.arange(len(labels2))+0.2, widths=0.4,
                     patch_artist=True, showmeans=True, meanline=True, medianprops={'linewidth': 2},
                     showfliers=False)

    # Customize Colors for data1
    colors1 = ['lightblue', 'lightgreen', 'pink', 'peachpuff', 'lavender',
               'salmon', 'skyblue', 'khaki', 'mediumpurple', 'hotpink']

    for patch, color in zip(bp1['boxes'], colors1):
        patch.set_facecolor(color)

    # Customize Colors for data2
    colors2 = ['darkblue', 'darkgreen', 'red', 'orange', 'purple',
               'brown', 'teal', 'olive', 'indigo', 'crimson']

    for patch, color in zip(bp2['boxes'], colors2):
        patch.set_facecolor(color)

    # Tweak Line Styles
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp1[element], color='gray', linewidth=1.5)
        plt.setp(bp2[element], color='gray', linewidth=1.5)

    # Customize Outlier ('flier') Style
    plt.setp(bp1['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
    plt.setp(bp2['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)

    # Customize Mean Marker (diamond)
    plt.setp(bp1['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)
    plt.setp(bp2['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)

    # Labels, Title, and Layout (Modified)
    ax.set_yscale('log')  # Set y-axis to logarithmic scale

    # Set y-axis limits and tick positions
    ylim_min = min(ax.get_ylim()[0], ax.get_ylim()[0])
    ylim_max = max(ax.get_ylim()[1], ax.get_ylim()[1])
    yticks = [1e-2, 1e-1, 1e0]
    yticklabels = ['$10^{-2}$', '$10^{-1}$', '$10^{0}$']

    ax.set_ylim(ylim_min, ylim_max)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.set_title(kwargs.get('title', 'Distribution of Distances to Centroids'), fontsize=14)
    ax.set_xlabel(kwargs.get('xlabel', 'Digit'), fontsize=12)
    ax.set_ylabel(kwargs.get('ylabel', 'Distance (Logarithmic Scale)'), fontsize=12)
    ax.set_xticks(range(len(labels1)))
    ax.set_xticklabels(labels1)

    ax.tick_params(axis='both', labelsize=10)

    # Create legend
    legend_labels = ['Data 1', 'Data 2']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors1[0]), plt.Rectangle((0, 0), 1, 1, facecolor=colors2[0])]
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)

    plt.tight_layout()

    # Saving and Displaying
    if save:
        filename = kwargs.get('filename', 'centroid_distances_comparison.png')
        plt.savefig(filename, dpi=300)

    plt.show()    

def plot_side_by_side_top_bottom(data1, data2, save=False, debug=False):
    """
    Plots side-by-side boxplots of distances to centroid for MNIST digits based on two datasets.

    Parameters:
    - data1: First dataset as a Numpy array, where the first column is distances and the second is labels.
    - data2: Second dataset as a Numpy array, similar to data1.
    - save: If True, saves the plot as a PNG file.
    - debug: If True, performs additional analysis on the lists (functionality not included here).
    """
    # Extract unique labels from the second column
    labels = np.unique(np.concatenate((data1[:, 1], data2[:, 1]))).astype(int)

    # Create lists to store distances for each label for both datasets
    distances_by_label_data1 = [data1[data1[:, 1] == label, 0] for label in labels]
    distances_by_label_data2 = [data2[data2[:, 1] == label, 0] for label in labels]

    if debug:
        # Placeholder for debugging function, which is not implemented in this snippet
        pass

    # Boxplot Creation with Styling
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Plot for data1
    bp1 = axs[0].boxplot(distances_by_label_data1, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True, medianprops={'linewidth': 2},
                         showfliers=False)

    # Plot for data2
    bp2 = axs[1].boxplot(distances_by_label_data2, labels=labels, patch_artist=True,
                         showmeans=True, meanline=True, medianprops={'linewidth': 2},
                         showfliers=False)

    # Define Colors for the plots
    colors = ['lightblue', 'lightgreen', 'pink', 'peachpuff', 'lavender',
              'salmon', 'skyblue', 'khaki', 'mediumpurple', 'hotpink']

    # Customize Colors for data1
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    # Customize Colors for data2
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)

    # Tweak Line Styles for both datasets
    for bp in [bp1, bp2]:
        for element in ['whiskers', 'caps', 'medians']:
            plt.setp(bp[element], color='gray', linewidth=1.5)
        plt.setp(bp['fliers'], marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=6)
        plt.setp(bp['means'], marker='D', markerfacecolor='white', markeredgecolor='black', markersize=8)

    # Labels, Title, and Layout (Modified)
    for ax in axs:
        ax.set_yscale('log')  # Set y-axis to logarithmic scale
        ax.set_ylabel('Distance (Logarithmic Scale)', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
    axs[1].set_xlabel('Digit', fontsize=12)

    # Title only for the first subplot
    axs[0].set_title('Distribution of Distances to Centroids for MNIST Digits - Dataset 1', fontsize=14)
    axs[1].set_title('Distribution of Distances to Centroids for MNIST Digits - Dataset 2', fontsize=14)

    plt.tight_layout()

    # Saving and Displaying
    if save:
        plt.savefig('centroid_distances_styled.png', dpi=300)
    plt.show()

# Example usage:
# Assuming 'data1' and 'data2' are your numpy arrays
# plot_centroid_mean_distance_boxplots_2(data1, data2, save=False, debug=False)


# Example usage:
# Assuming 'data' is your array with distances and labels
# plot_centroid_mean_distance_boxplots_2(data, save=True)



# Example usage:
# Assuming 'data' is your array with distances and labels
# plot_centroid_mean_distance_boxplots_2(data, save=True)

def analyze_lists_by_label(data):
    """
    Calculates descriptive statistics for lists within a list of lists, organized by label.

    Args:
        data (list of lists): List of lists where the index represents the label.

    Prints:
        A Pandas DataFrame containing descriptive statistics for each label.
    """
    stats_dict = {}  # Dictionary to store statistics by label

    # Calculate statistics for each list
    for label, values in enumerate(data):
        stats_dict[label] = {
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Standard Deviation': np.std(values),
            'Count': len(values)
        }

    # Create a Pandas DataFrame and display results
    df = pd.DataFrame.from_dict(stats_dict, orient='index')
    print(df)

def calculate_distances_to_centroids(train_correct_predictions, centroids):
    """
    Calculate the distances between the softmax outputs and their corresponding centroids.

    Args:
        train_correct_predictions (numpy.ndarray): Array of shape (59074, 12) containing softmax outputs
            and correct predictions for MNIST training images. Indexes 0 to 9 are the predictions,
            and index 10 is the correct prediction.
        centroids (numpy.ndarray): Array of shape (10, 10) containing cluster centroids. Row indexes
            correspond to the digit represented by the centroid.

    Returns:
        numpy.ndarray: Array of shape (59074, 2) where the first column contains the calculated distances,
            and the second column contains the correct predictions.
    """
    results = np.zeros((train_correct_predictions.shape[0], 2))

    for i, row in enumerate(train_correct_predictions):
        # Get the correct prediction (label) for this row
        correct_prediction = int(row[10])

        # Calculate the distance between the softmax outputs (row[:10]) and the corresponding centroid
        distance = np.linalg.norm(row[:10] - centroids[correct_prediction])

        # Store the distance and correct prediction in the results array
        results[i, 0] = distance
        results[i, 1] = correct_prediction

    return results      

# Softmax output averages for each digit
import numpy as np
import matplotlib.pyplot as plt

def plot_digit_averages(train_correct_predictions):
    # Get the unique labels (digits) from column 11
    labels = np.unique(train_correct_predictions[:, 11]).astype(int)
    
    # Create a figure and subplots for each digit
    fig, axs = plt.subplots(1, len(labels), figsize=(20, 5), sharey=True)
    
    # Iterate over each digit
    for i, label in enumerate(labels):
        # Get the predictions for the current digit
        digit_predictions = train_correct_predictions[train_correct_predictions[:, 11] == label, :10]
        
        # Calculate the average value for each index
        averages = np.mean(digit_predictions, axis=0)
        
        # Plot the bar graph for the current digit
        axs[i].bar(np.arange(10), averages)
        
        # Set the y-axis to logarithmic scale
        axs[i].set_yscale('log')
        
        # Set the title and labels for the current subplot
        axs[i].set_title(f'Digit {label}')
        axs[i].set_xlabel('Digit Index')
        axs[i].set_ylabel('Average Value (Logarithmic)')
        
        # Set the x-tick positions and labels
        axs[i].set_xticks(np.arange(10))
        axs[i].set_xticklabels(np.arange(10))
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    
    # Display the plot
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_digit_averages(train_correct_predictions, train_incorrect_predictions, color1='skyblue', color2='lightcoral'):
    # Get the unique labels (digits) from column 11
    labels = np.unique(train_correct_predictions[:, 10]).astype(int)

    # Create a figure and subplots for each digit (2 rows: correct and incorrect predictions)
    fig, axs = plt.subplots(2, len(labels), figsize=(20, 10))

    # Plot correct predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current digit
        digit_predictions = train_correct_predictions[train_correct_predictions[:, 10] == label, :10]

        # Calculate the average value for each index
        averages = np.mean(digit_predictions, axis=0)

        # Plot the bar graph for the current digit
        axs[0, i].bar(np.arange(10), averages, color=color1)

        # Set the y-axis to logarithmic scale
        axs[0, i].set_yscale('log')

        # Set the y-axis limits to start from 10^-4
        axs[0, i].set_ylim(bottom=1e-4)

        # Set the title and labels for the current subplot
        axs[0, i].set_title(f'Digit {label} (Correct)')
        axs[0, i].set_xlabel('Digit Index')
        axs[0, i].set_ylabel('Average Softmax Value (Logarithmic)')

        # Set the x-tick positions and labels
        axs[0, i].set_xticks(np.arange(10))
        axs[0, i].set_xticklabels(np.arange(10))

    # Plot incorrect predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current digit
        digit_predictions = train_incorrect_predictions[train_incorrect_predictions[:, 11] == label, :10]

        # Calculate the average value for each index
        averages = np.mean(digit_predictions, axis=0)

        # Plot the bar graph for the current digit
        axs[1, i].bar(np.arange(10), averages, color=color2)

        # Set the y-axis to logarithmic scale
        axs[1, i].set_yscale('log')

        # Set the y-axis limits to start from 10^-4
        axs[1, i].set_ylim(bottom=1e-4)

        # Set the title and labels for the current subplot
        axs[1, i].set_title(f'Digit {label} (Incorrect)')
        axs[1, i].set_xlabel('Digit Index')
        axs[1, i].set_ylabel('Average Softmax Value (Logarithmic)')

        # Set the x-tick positions and labels
        axs[1, i].set_xticks(np.arange(10))
        axs[1, i].set_xticklabels(np.arange(10))

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()