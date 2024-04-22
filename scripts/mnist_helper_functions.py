import torch
import numpy as np
from scipy.optimize import curve_fit

###########
# PROCESS #
###########

def linear_fit(x, a, b):
    return a * x + b

def sanity_check(data, num_cols=10, pred_col=11):
    """
    Perform a sanity check on the input array.

    Args:
        data (numpy.ndarray): Array of shape (n, m) where the first num_cols columns contain softmax probabilities,
                              and the pred_col column contains the predicted labels.
        num_cols (int, optional): Number of columns containing softmax probabilities. Default is 10.
        pred_col (int, optional): Index of the column containing predicted labels. Default is 11.

    Raises:
        AssertionError: If the predicted label does not correspond to the highest softmax probability.
    """
    num_samples = data.shape[0]

    for i in range(num_samples):
        softmax_probs = data[i, :num_cols]
        predicted_label = int(data[i, pred_col])

        # Find the index of the highest softmax probability
        max_prob_index = np.argmax(softmax_probs)

        # Assert that the predicted label corresponds to the highest softmax probability
        assert max_prob_index == predicted_label, f"Sample {i}: Predicted label {predicted_label} does not correspond to the highest softmax probability index {max_prob_index}"

    print("Sanity check passed!")

def logits_to_softmax(logits, num_cols=10):
    """
    Convert logits to softmax probabilities.

    Args:
        logits (numpy.ndarray): Array of shape (n, m) where the first num_cols columns contain the logits.
        num_cols (int, optional): Number of columns to be converted to softmax probabilities. Default is 10.

    Returns:
        numpy.ndarray: Array of shape (n, m) where the first num_cols columns contain the softmax probabilities.
    """
    # Extract the logits (first num_cols columns)
    logits_values = logits[:, :num_cols]

    # Calculate the softmax probabilities
    exp_logits = np.exp(logits_values)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Create a new array to store the result
    result = np.zeros_like(logits)

    # Copy the softmax probabilities to the first num_cols columns of the result
    result[:, :num_cols] = softmax_probs

    # Copy the remaining columns from the input to the result
    result[:, num_cols:] = logits[:, num_cols:]

    return result

def display_misclassifications_side_by_side_cifar10(x1, x2, y_true, y_pred, class_labels, w=20, h=10, save=False):
    """
    Displays heatmaps of the confusion matrices for two CIFAR-10 datasets side by side and example images of misclassified objects.

    Parameters:
    - x1: Numpy array of CIFAR-10 images for the first dataset.
    - x2: Numpy array of CIFAR-10 images for the second dataset.
    - y_true: Numpy array containing the true labels for both datasets.
    - y_pred: Numpy array containing the predicted labels for both datasets.
    - class_labels: List of class labels for the CIFAR-10 dataset.
    - w: Width of the figure. Defaults to 20.
    - h: Height of the figure. Defaults to 10.
    - save: Boolean, if True, saves the confusion matrix plot. Defaults to False.

    Returns:
    - The confusion matrices for both datasets.

    Example:
    conf_matrix1, conf_matrix2 = display_misclassifications_side_by_side_cifar10(x1_test, x2_test, y_test, y_pred, class_labels, save=True)
    """
    # Retrieve data
    true_labels1 = x1[:, y_true]
    predicted_labels1 = x1[:, y_pred]
    true_labels2 = x2[:, y_true]
    predicted_labels2 = x2[:, y_pred]

    # Compute the confusion matrices for both datasets
    conf_matrix1 = confusion_matrix(true_labels1, predicted_labels1)
    conf_matrix2 = confusion_matrix(true_labels2, predicted_labels2)

    # Create a mask for non-zero values
    non_zero_mask1 = np.array(conf_matrix1 > 0, dtype=int)
    non_zero_mask2 = np.array(conf_matrix2 > 0, dtype=int)

    plt.figure(figsize=(w, h))

    # Plotting for the first dataset
    plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
    sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Reds', mask=non_zero_mask1 == 0, xticklabels=class_labels, yticklabels=class_labels)
    sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', mask=non_zero_mask1, cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix for CIFAR-10 Training Dataset')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)

    # Plotting for the second dataset
    plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
    sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Reds', mask=non_zero_mask2 == 0, xticklabels=class_labels, yticklabels=class_labels)
    sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', mask=non_zero_mask2, cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix for CIFAR-10 Testing Dataset')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)

    # Decrease the padding between the plots
    plt.subplots_adjust(wspace=0.1)

    if save:
        plt.savefig('cifar10_combined_confusion_matrix.png')

    plt.show()

    return conf_matrix1, conf_matrix2

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

def plot_accuracy_decrements(results, results_overall, save=True):
    """
    Plots the accuracy vs threshold decrement for each digit class and the overall accuracy.

    Parameters:
        results (numpy.ndarray): The results array obtained from the calculate_accuracy_decrements function.
                                 Expected columns:
                                 - Column 0: Total number of values below the threshold
                                 - Column 1: Current threshold
                                 - Column 2: Decrement
                                 - Column 3: Digit class
                                 - Column 4: Digit class prediction accuracy
                                 - Column 5: Total values (correct + incorrect) for the digit class

        results_overall (numpy.ndarray): The results array obtained from the calculate_accuracy_decrements_overall function.
                                         Expected columns:
                                         - Column 0: Total number of values below the threshold (overall)
                                         - Column 1: Decrement
                                         - Column 2: Overall accuracy
                                         - Column 3: Total correct predictions (overall)
                                         - Column 4: Total values (overall)

    Returns:
        None
    """
    # Define column indices
    COL_DECREMENT = 2
    COL_DIGIT_CLASS = 3
    COL_ACCURACY = 4

    COL_OVERALL_DECREMENT = 1
    COL_OVERALL_ACCURACY = 2

    # Get the unique digit classes
    digit_classes = np.unique(results[:, COL_DIGIT_CLASS])

    # Create a figure and subplots for each digit class and overall accuracy
    fig, axs = plt.subplots(1, len(digit_classes) + 1, figsize=(24, 5))

    # Iterate over each unique digit class
    for i, digit_class in enumerate(digit_classes):
        # Find the rows corresponding to the current digit class
        digit_rows = results[results[:, COL_DIGIT_CLASS] == digit_class]

        # Extract the accuracy and decrement values for the current digit class
        accuracy = digit_rows[:, COL_ACCURACY]
        decrement = digit_rows[:, COL_DECREMENT]

        # Plot the accuracy vs decrement for the current digit class
        axs[i].plot(decrement, accuracy, marker='o', color='lightgreen')
        axs[i].set_title(f'{int(digit_class)}')
        axs[i].set_ylim(0.8, 1)
        axs[i].grid(True)

        # Add vertical lines for each decrement
        for dec in decrement:
            axs[i].axvline(x=dec, color='gray', linestyle='--', linewidth=0.5)

        # Show y-axis labels only on the first subplot
        if i == 0:
            axs[i].set_ylabel('Accuracy')
        else:
            axs[i].set_ylabel('')

    # Extract the overall accuracy and decrement values
    overall_accuracy = results_overall[:, COL_OVERALL_ACCURACY]
    overall_decrement = results_overall[:, COL_OVERALL_DECREMENT]

    # Plot the overall accuracy
    axs[-1].plot(overall_decrement, overall_accuracy, marker='o', color='lightgreen')
    axs[-1].set_title('Overall')
    axs[-1].set_ylim(0.8, 1)
    axs[-1].grid(True)

    # Add vertical lines for each decrement in the overall accuracy plot
    for dec in overall_decrement:
        axs[-1].axvline(x=dec, color='gray', linestyle='--', linewidth=0.5)

    # Set the x-axis label for the entire figure
    fig.text(0.5, 0.02, 'Threshold Decrement', ha='center')

    # Set the title for the entire figure
    fig.suptitle('Class Prediction Accuracy vs Distance to Threshold Decrement')

    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    # Save the plot as an image file
    if save:
        plt.savefig('plot_accuracy_decrements.png')

    # Display the plot
    plt.show()

import matplotlib.pyplot as plt

# def single_plot_accuracy_decrements(results, results_overall, save=True):
#     """
#     Plots the accuracy vs threshold decrement for each digit class and the overall accuracy on a single plot.

#     Parameters:
#         results (numpy.ndarray): The results array obtained from the calculate_accuracy_decrements function.
#                                  Expected columns:
#                                  - Column 0: Total number of values below the threshold
#                                  - Column 1: Current threshold
#                                  - Column 2: Decrement
#                                  - Column 3: Digit class
#                                  - Column 4: Digit class prediction accuracy
#                                  - Column 5: Total values (correct + incorrect) for the digit class

#         results_overall (numpy.ndarray): The results array obtained from the calculate_accuracy_decrements_overall function.
#                                          Expected columns:
#                                          - Column 0: Total number of values below the threshold (overall)
#                                          - Column 1: Decrement
#                                          - Column 2: Overall accuracy
#                                          - Column 3: Total correct predictions (overall)
#                                          - Column 4: Total values (overall)

#         save (bool): Whether to save the plot as an image file. Default is True.

#     Returns:
#         None
#     """
#     # Define column indices
#     COL_DECREMENT = 2
#     COL_DIGIT_CLASS = 3
#     COL_ACCURACY = 4

#     COL_OVERALL_DECREMENT = 1
#     COL_OVERALL_ACCURACY = 2

#     # Get the unique digit classes
#     digit_classes = np.unique(results[:, COL_DIGIT_CLASS])

#     # Create a figure and a single axis
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Define color map for different colors
#     cmap = plt.cm.get_cmap('viridis', len(digit_classes) + 1)

#     # Plot accuracy vs decrement for each digit class
#     for i, digit_class in enumerate(digit_classes):
#         digit_rows = results[results[:, COL_DIGIT_CLASS] == digit_class]
#         accuracy = digit_rows[:, COL_ACCURACY]
#         decrement = digit_rows[:, COL_DECREMENT]
#         ax.plot(decrement, accuracy, marker='o', color=cmap(i), label=f'Digit {int(digit_class)}')

#     # Plot overall accuracy vs decrement
#     overall_accuracy = results_overall[:, COL_OVERALL_ACCURACY]
#     overall_decrement = results_overall[:, COL_OVERALL_DECREMENT]
#     ax.plot(overall_decrement, overall_accuracy, marker='o', color=cmap(len(digit_classes)), label='Overall')

#     # Set the x-axis and y-axis labels
#     ax.set_xlabel('Threshold Decrement')
#     ax.set_ylabel('Accuracy')

#     # Set the y-axis limits and ticks
#     ax.set_ylim(0.84, 1)
#     ax.set_yticks(np.arange(0.84, 1.01, 0.02))

#     # Add horizontal lines for each y-tick
#     ax.grid(True, which='both', axis='y', linewidth=0.5, linestyle='--', alpha=0.7)

#     # Add vertical lines for each decrement value
#     for dec in overall_decrement:
#         ax.axvline(x=dec, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

#     # Add labels for each decrement value below the x-axis
#     ax.set_xticks(overall_decrement)
#     ax.set_xticklabels([f'{dec:.1f}' for dec in overall_decrement])

#     # Set the title for the plot
#     ax.set_title('Class Prediction Accuracy vs Distance to Threshold Decrement')

#     # Add a legend on the bottom left
#     ax.legend(loc='lower left', ncol=2)

#     # Adjust the spacing
#     plt.tight_layout()

#     # Save the plot as an image file
#     if save:
#         plt.savefig('single_plot_accuracy_decrements.png')

#     # Display the plot
#     plt.show()

def single_plot_accuracy_decrements(results, results_overall, labels=None, dataset="MNIST", save=True):
    """
    Plots the accuracy vs threshold decrement for each digit class and the overall accuracy on a single plot.

    Parameters:
        results (numpy.ndarray): The results array obtained from the calculate_accuracy_decrements function.
                                 Expected columns:
                                 - Column 0: Total number of values below the threshold
                                 - Column 1: Current threshold
                                 - Column 2: Decrement
                                 - Column 3: Digit class
                                 - Column 4: Digit class prediction accuracy
                                 - Column 5: Total values (correct + incorrect) for the digit class
        results_overall (numpy.ndarray): The results array obtained from the calculate_accuracy_decrements_overall function.
                                         Expected columns:
                                         - Column 0: Total number of values below the threshold (overall)
                                         - Column 1: Decrement
                                         - Column 2: Overall accuracy
                                         - Column 3: Total correct predictions (overall)
                                         - Column 4: Total values (overall)
        labels (list): List of labels for each digit class. Default is None.
        dataset (str): Name of the dataset. Default is "MNIST".
        save (bool): Whether to save the plot as an image file. Default is True.

    Returns:
        None
    """
    # Define column indices
    COL_DECREMENT = 2
    COL_DIGIT_CLASS = 3
    COL_ACCURACY = 4
    COL_OVERALL_DECREMENT = 1
    COL_OVERALL_ACCURACY = 2

    # Get the unique digit classes
    digit_classes = np.unique(results[:, COL_DIGIT_CLASS])

    # Create a figure and a single axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define color map for different colors
    cmap = plt.cm.get_cmap('viridis', len(digit_classes) + 1)

    # Plot accuracy vs decrement for each digit class
    for i, digit_class in enumerate(digit_classes):
        digit_rows = results[results[:, COL_DIGIT_CLASS] == digit_class]
        accuracy = digit_rows[:, COL_ACCURACY]
        decrement = digit_rows[:, COL_DECREMENT]
        if labels is not None:
            label = labels[int(digit_class)]
        else:
            label = f'Digit {int(digit_class)}'
        ax.plot(decrement, accuracy, marker='o', color=cmap(i), label=label)

    # Plot overall accuracy vs decrement
    overall_accuracy = results_overall[:, COL_OVERALL_ACCURACY]
    overall_decrement = results_overall[:, COL_OVERALL_DECREMENT]
    ax.plot(overall_decrement, overall_accuracy, marker='o', color=cmap(len(digit_classes)), label='Overall')

    # Set the x-axis and y-axis labels
    ax.set_xlabel('Threshold Decrement')
    ax.set_ylabel('Accuracy')

    # Set the y-axis limits and ticks
    ax.set_ylim(0.84, 1)
    ax.set_yticks(np.arange(0.84, 1.01, 0.02))

    # Add horizontal lines for each y-tick
    ax.grid(True, which='both', axis='y', linewidth=0.5, linestyle='--', alpha=0.7)

    # Add vertical lines for each decrement value
    for dec in overall_decrement:
        ax.axvline(x=dec, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    # Add labels for each decrement value below the x-axis
    ax.set_xticks(overall_decrement)
    ax.set_xticklabels([f'{dec:.1f}' for dec in overall_decrement])

    # Set the title for the plot
    ax.set_title(f'{dataset} Class Prediction Accuracy vs Distance to Threshold Decrement')

    # Add a legend on the bottom left
    ax.legend(loc='lower left', ncol=2)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as an image file
    if save:
        plt.savefig('single_plot_accuracy_decrements.png')

    # Display the plot
    plt.show()

def centroid_distance_overlap_latex(d2c_train_correct, d2c_train_incorrect, d2c_test_correct, d2c_test_incorrect):
    """
    Compares the distances in column 1 of d2c_train_correct, d2c_train_incorrect, d2c_test_correct, and d2c_test_incorrect
    for each label (digit) in column 2, and calculates the count and percentage of rows
    in d2c_train_correct and d2c_test_correct where column 1 has values greater than or equal to the
    corresponding values in d2c_train_incorrect and d2c_test_incorrect.
    Args:
    d2c_train_correct (numpy.ndarray): Array of shape (n, 2) containing distances and labels for correct predictions in training data.
    d2c_train_incorrect (numpy.ndarray): Array of shape (m, 2) containing distances and labels for incorrect predictions in training data.
    d2c_test_correct (numpy.ndarray): Array of shape (p, 2) containing distances and labels for correct predictions in testing data.
    d2c_test_incorrect (numpy.ndarray): Array of shape (q, 2) containing distances and labels for incorrect predictions in testing data.

    Returns:
    None
    """

    # Get the unique labels (digits) from column 2
    labels = np.unique(d2c_train_correct[:, 1]).astype(int)

    # Initialize arrays to store the results
    train_count_greater_equal = np.zeros(len(labels), dtype=int)
    train_total_count = np.zeros(len(labels), dtype=int)
    test_train_count_greater_equal = np.zeros(len(labels), dtype=int)
    test_train_total_count = np.zeros(len(labels), dtype=int)

    # Iterate over each label (digit)
    for i, label in enumerate(labels):
        # Get the distances for the current label from training arrays
        train_correct_distances = d2c_train_correct[d2c_train_correct[:, 1] == label, 0]
        train_incorrect_distances = d2c_train_incorrect[d2c_train_incorrect[:, 1] == label, 0]
        
        # Get the distances for the current label from testing arrays
        test_correct_distances = d2c_test_correct[d2c_test_correct[:, 1] == label, 0]
        
        # Count the number of rows in d2c_train_correct where column 1 has values greater than or equal to the minimum value in d2c_train_incorrect
        train_count_greater_equal[i] = np.sum(train_correct_distances >= np.min(train_incorrect_distances))
        
        # Count the total number of rows for the current label in d2c_train_correct
        train_total_count[i] = len(train_correct_distances)
        
        # Count the number of rows in d2c_train_correct where column 1 has values greater than or equal to the minimum value in d2c_test_incorrect
        test_train_count_greater_equal[i] = np.sum(test_correct_distances >= np.min(train_incorrect_distances))
        
        # Count the total number of rows for the current label in d2c_train_correct
        test_train_total_count[i] = len(test_correct_distances)

    # Calculate the percentage of rows greater than or equal for each label in training data
    train_percentage_greater_equal = train_count_greater_equal / train_total_count * 100

    # Calculate the percentage of rows greater than or equal for each label between d2c_train_correct and d2c_test_incorrect
    test_train_percentage_greater_equal = test_train_count_greater_equal / test_train_total_count * 100

    # Calculate the totals for each column
    train_count_total = np.sum(train_count_greater_equal)
    train_total = np.sum(train_total_count)
    train_percentage_total = train_count_total / train_total * 100

    test_train_count_total = np.sum(test_train_count_greater_equal)
    test_train_total = np.sum(test_train_total_count)
    test_train_percentage_total = test_train_count_total / test_train_total * 100

    # Create the LaTeX table
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("Digit & \\multicolumn{3}{c|}{Train} & \\multicolumn{3}{c|}{Test-Train} \\\\")
    print("\\hline")
    print(" & Cnt & Ttl & Opct & Cnt & Ttl & Opct \\\\")
    print("\\hline")
    for i, label in enumerate(labels):
        print(f"{label} & {train_count_greater_equal[i]} & {train_total_count[i]} & {train_percentage_greater_equal[i]:.2f}\\% & {test_train_count_greater_equal[i]} & {test_train_total_count[i]} & {test_train_percentage_greater_equal[i]:.2f}\\% \\\\")
    print("\\hline")
    print(f"Totals & {train_count_total} & {train_total} & {train_percentage_total:.2f}\\% & {test_train_count_total} & {test_train_total} & {test_train_percentage_total:.2f}\\% \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Comparison of centroid distances between correct and incorrect predictions}")
    print("\\label{tab:centroid_distance_overlap}")
    print("\\end{table}")

def plot_digit_averages(train_correct_predictions, train_incorrect_predictions, color1='skyblue', color2='lightcoral', data="Training Data"):
    # Get the unique labels (digits) from column 11
    labels = np.unique(train_correct_predictions[:, 10]).astype(int)

    # Create a figure and subplots for each digit (2 rows: correct and incorrect predictions)
    fig, axs = plt.subplots(2, len(labels), figsize=(20, 10))
    fig.suptitle(f'{data} - Softmax Average Distributions for Correct and Incorrect Digit Predictions', fontsize=16)

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
        # Set the title for the current subplot
        axs[0, i].set_title(f'Digit {label} (Correct)', fontsize=12)
        # Set the x-tick positions and labels
        axs[0, i].set_xticks(np.arange(10))
        axs[0, i].set_xticklabels(np.arange(10), fontsize=10)
        # Add x-axis grid lines
        axs[0, i].set_xticks(np.arange(10), minor=True)
        axs[0, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add y-axis grid lines
        axs[0, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

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
        # Set the title for the current subplot
        axs[1, i].set_title(f'Digit {label} (Incorrect)', fontsize=12)
        # Set the x-tick positions and labels
        axs[1, i].set_xticks(np.arange(10))
        axs[1, i].set_xticklabels(np.arange(10), fontsize=10)
        # Add x-axis grid lines
        axs[1, i].set_xticks(np.arange(10), minor=True)
        axs[1, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        # Add y-axis grid lines
        axs[1, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Digit Index', ha='center', fontsize=14)

    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=14)

    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing for the main title

    # Display the plot
    plt.show()    

def plot_centroid_distance_bars(train_correct_predictions, train_incorrect_predictions, color1='skyblue', color2='lightcoral', data="Training Data"):
    # Get the unique labels (classes) from column 11
    labels = np.unique(train_correct_predictions[:, 10]).astype(int)
    
    # Create a figure and subplots for each class (2 rows: correct and incorrect predictions)
    fig, axs = plt.subplots(2, len(labels), figsize=(20, 10))
    fig.suptitle(f'{data} - Softmax Average Distributions for Correct and Incorrect Class Predictions', fontsize=16)
    
    # Plot correct predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current class
        class_predictions = train_correct_predictions[train_correct_predictions[:, 10] == label, :10]
        
        # Calculate the average value for each index
        averages = np.mean(class_predictions, axis=0)
        
        # Plot the bar graph for the current class
        axs[0, i].bar(np.arange(10), averages, color=color1)
        
        # Set the y-axis to logarithmic scale
        axs[0, i].set_yscale('log')
        
        # Set the y-axis limits to start from 10^-4
        axs[0, i].set_ylim(bottom=1e-4)
        
        # Set the title for the current subplot
        axs[0, i].set_title(f'Class {label} (Correct)', fontsize=12)
        
        # Set the x-tick positions and labels
        axs[0, i].set_xticks(np.arange(10))
        axs[0, i].set_xticklabels(np.arange(10), fontsize=10)
        
        # Add x-axis grid lines
        axs[0, i].set_xticks(np.arange(10), minor=True)
        axs[0, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add y-axis grid lines
        axs[0, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot incorrect predictions
    for i, label in enumerate(labels):
        # Get the predictions for the current class
        class_predictions = train_incorrect_predictions[train_incorrect_predictions[:, 11] == label, :10]
        
        # Calculate the average value for each index
        averages = np.mean(class_predictions, axis=0)
        
        # Plot the bar graph for the current class
        axs[1, i].bar(np.arange(10), averages, color=color2)
        
        # Set the y-axis to logarithmic scale
        axs[1, i].set_yscale('log')
        
        # Set the y-axis limits to start from 10^-4
        axs[1, i].set_ylim(bottom=1e-4)
        
        # Set the title for the current subplot
        axs[1, i].set_title(f'Class {label} (Incorrect)', fontsize=12)
        
        # Set the x-tick positions and labels
        axs[1, i].set_xticks(np.arange(10))
        axs[1, i].set_xticklabels(np.arange(10), fontsize=10)
        
        # Add x-axis grid lines
        axs[1, i].set_xticks(np.arange(10), minor=True)
        axs[1, i].grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Add y-axis grid lines
        axs[1, i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set x-axis label at the bottom of the figure
    fig.text(0.5, 0.04, 'Class Index', ha='center', fontsize=14)
    
    # Set y-axis label on the left side of the figure
    fig.text(0.04, 0.5, 'Average Softmax Value (Logarithmic)', va='center', rotation='vertical', fontsize=14)
    
    # Adjust the spacing between subplots
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    fig.subplots_adjust(top=0.9)  # Adjust the top spacing for the main title
    
    # Display the plot
    plt.show()    

def calculate_distances_to_centroids(train_correct_predictions, centroids):
    """
    Calculates the distances to the centroids for each row in the given numpy array.

    Parameters:
    - train_correct_predictions: A numpy array of shape (n, 12) where:
        - The first 10 columns represent the output of the softmax function.
        - Column 11 represents the class label.
        - Column 12 represents the predicted class.
    - centroids: A numpy array of shape (10, 10) where:
        - The row index represents the class label.
        - The columns represent the centroid values.

    Returns:
    - distances_to_centroids: A numpy array of shape (n, 2) where:
        - The first column represents the distance to the centroid.
        - The second column represents the class label.
    """
    n = train_correct_predictions.shape[0]
    distances_to_centroids = np.zeros((n, 2))

    for i in range(n):
        row = train_correct_predictions[i]
        class_label = int(row[10])
        centroid = centroids[class_label]
        distance = np.linalg.norm(row[:10] - centroid)
        
        distances_to_centroids[i, 0] = distance
        distances_to_centroids[i, 1] = class_label

    return distances_to_centroids

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

    # if debug:
    #     analyze_lists_by_label(distances_by_label1)
    #     analyze_lists_by_label(distances_by_label2)
    #     analyze_lists_by_label(distances_by_label3)
    #     analyze_lists_by_label(distances_by_label4)

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
    ax1.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_yscale('log')
    ax2.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)

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
    ax1.set_xticks(range(len(labels1)), minor=True)
    ax1.grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.set_xticks(range(len(labels3)))
    ax2.set_xticklabels(labels3)
    ax2.set_xticks(range(len(labels3)), minor=True)
    ax2.grid(True, which='minor', axis='x', linestyle='--', linewidth=0.5, alpha=0.7)

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

def calculate_class_accuracies(train_np, num_classes=10, class_label_col=10, class_pred_col=11):
    """
    Calculates the prediction accuracy for each class in the given numpy array.

    Parameters:
    - train_np: A numpy array of shape (n, m) where:
        - The first m-2 columns represent the output of the softmax function.
        - Column with index 'class_label_col' represents the class label.
        - Column with index 'class_pred_col' represents the class prediction.
    - num_classes: The number of distinct classes (default: 10).
    - class_label_col: The index of the column containing the class labels (default: 10).
    - class_pred_col: The index of the column containing the class predictions (default: 11).

    Returns:
    - A numpy array of shape (1, num_classes) where:
        - The column index represents the class.
        - The value represents the prediction accuracy for the class.
    """
    class_accuracies = np.zeros((1, num_classes))

    for class_label in range(num_classes):
        class_rows = train_np[train_np[:, class_label_col] == class_label]
        total_rows = class_rows.shape[0]
        correct_predictions = np.sum(class_rows[:, class_label_col] == class_rows[:, class_pred_col])
        
        if total_rows > 0:
            accuracy = correct_predictions / total_rows
        else:
            accuracy = 0.0
        
        class_accuracies[0, class_label] = accuracy

    return class_accuracies           

def plot_accuracy_vs_distance_linear_fit(train_distance, train_accuracy, test_distance, test_accuracy, save=True):
    """
    Plots accuracy against mean class distance to centroid for both training and testing data,
    with a linear fit for the trend in the data.
    
    Parameters:
    - train_distance: A numpy array with shape (10,) containing mean distances to centroid for each class in the training set.
    - train_accuracy: A numpy array with shape (1,10) containing accuracies for each class in the training set.
    - test_distance: A numpy array with shape (10,) containing mean distances to centroid for each class in the testing set.
    - test_accuracy: A numpy array with shape (1,10) containing accuracies for each class in the testing set.
    - save: A boolean indicating whether to save the plot. Default is True.
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
    plt.figure(figsize=(14, 8))
    
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
    
    # Add horizontal lines
    plt.axhline(y=0.96, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(y=0.97, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(y=0.98, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(y=0.99, color='gray', linestyle='--', linewidth=0.8)
    
    # Set the title
    plt.title('MNIST Classification: Train vs Test Accuracy and Mean Distance to Centroid with Linear Fit')
    
    # Show the grid
    plt.grid(True)
    
    # Show the legend with fit functions
    plt.legend(loc='lower left')
    
    # Show the plot
    plt.show()
    
    if save:
        plt.savefig('MNIST_Classification_Train_Test_Accuracy_Mean_Distance_to_Centroid_Linear_Fit.png')    

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
    ax1.legend(loc='center left')  # ax1.legend()
    
    # Add horizontal lines for training data
    ax1.axhline(y=np.mean(training_correct_distances), color='skyblue', linestyle='--', linewidth=1)
    ax1.axhline(y=np.mean(training_incorrect_distances), color='lightcoral', linestyle='--', linewidth=1)
    
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
    ax2.legend(loc='center left')  # ax2.legend()
    
    # Add horizontal lines for testing data
    ax2.axhline(y=np.mean(testing_correct_distances), color='lightgreen', linestyle='--', linewidth=1)
    ax2.axhline(y=np.mean(testing_incorrect_distances), color='lightcoral', linestyle='--', linewidth=1)
    
    # Annotate each bar with the mean distance value for testing data
    for i, (correct_distance, incorrect_distance) in enumerate(zip(testing_correct_distances, testing_incorrect_distances)):
        ax2.text(i - bar_width/2, correct_distance, f'{correct_distance:.4f}', ha='center', va='bottom')
        ax2.text(i + bar_width/2, incorrect_distance, f'{incorrect_distance:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    if save:
        plt.savefig('Combined_mean_distances_double_bars.png')        

def calculate_accuracy_decrements(d2c_test_correct, d2c_test_incorrect, lowest_values):
    # Get the unique digit classes
    digit_classes = np.unique(d2c_test_correct[:, 1])

    # Initialize an array to store the results
    results = np.zeros((100, 6))

    # Iterate over each unique digit class
    for digit_class in digit_classes:
        # Find the indices of rows with the current digit class in d2c_test_correct
        indices_correct = np.where(d2c_test_correct[:, 1] == digit_class)
        class_rows_correct = d2c_test_correct[indices_correct]

        # Find the indices of rows with the current digit class in d2c_test_incorrect
        indices_incorrect = np.where(d2c_test_incorrect[:, 1] == digit_class)
        class_rows_incorrect = d2c_test_incorrect[indices_incorrect]

        # Get the number of rows for the current digit class in d2c_test_correct and d2c_test_incorrect
        num_correct = class_rows_correct.shape[0]
        num_incorrect = class_rows_incorrect.shape[0]

        # Get the threshold value for the current digit class from lowest_values
        threshold = lowest_values[int(digit_class), 0]

        # Iterate over decrements [0.0, 0.1, 0.2, ..., 0.9]
        for i, decrement in enumerate(np.arange(0, 1, 0.1)):
            # Calculate the current threshold with the decrement
            current_threshold = threshold * (1 - decrement)

            # Count the number of values below the current threshold in class_rows_correct
            count_below_threshold_correct = np.sum(class_rows_correct[:, 0] < current_threshold)

            # Count the number of values below the current threshold in class_rows_incorrect
            count_below_threshold_incorrect = np.sum(class_rows_incorrect[:, 0] < current_threshold)

            # Calculate the total number of values below the threshold
            total_below_threshold = count_below_threshold_correct + count_below_threshold_incorrect

            # Calculate the digit class prediction accuracy
            accuracy = count_below_threshold_correct / (num_correct + num_incorrect)

            # Calculate the index in the results array
            index = int(digit_class * 10 + i)

            # Store the results in the array
            results[index, 0] = total_below_threshold
            results[index, 1] = current_threshold
            results[index, 2] = decrement
            results[index, 3] = digit_class
            results[index, 4] = accuracy
            results[index, 5] = num_correct + num_incorrect  # Total values

    return results


def calculate_accuracy_decrements_overall(d2c_test_correct, d2c_test_incorrect, lowest_values):
    # Get the unique digit classes
    digit_classes = np.unique(d2c_test_correct[:, 1])

    # Initialize an array to store the results
    results = np.zeros((10, 5))

    # Iterate over decrements [0.0, 0.1, 0.2, ..., 0.9]
    for i, decrement in enumerate(np.arange(0, 1, 0.1)):
        # Initialize variables to store the totals for each decrement
        total_below_threshold = 0
        total_correct = 0
        total_values = 0

        # Iterate over each unique digit class
        for digit_class in digit_classes:
            # Find the indices of rows with the current digit class in d2c_test_correct
            indices_correct = np.where(d2c_test_correct[:, 1] == digit_class)
            class_rows_correct = d2c_test_correct[indices_correct]

            # Find the indices of rows with the current digit class in d2c_test_incorrect
            indices_incorrect = np.where(d2c_test_incorrect[:, 1] == digit_class)
            class_rows_incorrect = d2c_test_incorrect[indices_incorrect]

            # Get the number of rows for the current digit class in d2c_test_correct and d2c_test_incorrect
            num_correct = class_rows_correct.shape[0]
            num_incorrect = class_rows_incorrect.shape[0]

            # Get the threshold value for the current digit class from lowest_values
            threshold = lowest_values[int(digit_class), 0]

            # Calculate the current threshold with the decrement
            current_threshold = threshold * (1 - decrement)

            # Count the number of values below the current threshold in class_rows_correct
            count_below_threshold_correct = np.sum(class_rows_correct[:, 0] < current_threshold)

            # Count the number of values below the current threshold in class_rows_incorrect
            count_below_threshold_incorrect = np.sum(class_rows_incorrect[:, 0] < current_threshold)

            # Update the totals for the current decrement
            total_below_threshold += count_below_threshold_correct + count_below_threshold_incorrect
            total_correct += count_below_threshold_correct
            total_values += num_correct + num_incorrect

        # Calculate the overall accuracy for the current decrement
        accuracy = total_correct / total_values

        # Store the results in the array
        results[i, 0] = total_below_threshold
        results[i, 1] = decrement
        results[i, 2] = accuracy
        results[i, 3] = total_correct
        results[i, 4] = total_values

    return results

def find_lowest_values(d2c_train_incorrect):
    # Initialize an array to store the lowest values for each class
    lowest_values = np.zeros((10, 2))

    # Get the unique digit classes
    digit_classes = np.unique(d2c_train_incorrect[:, 1])

    # Iterate over each unique digit class
    for digit_class in digit_classes:
        # Find the indices of rows with the current digit class
        indices = np.where(d2c_train_incorrect[:, 1] == digit_class)

        # Extract the rows corresponding to the current digit class
        class_rows = d2c_train_incorrect[indices]

        # Find the row with the minimum distance for the current digit class
        min_row = class_rows[np.argmin(class_rows[:, 0])]

        # Update the lowest value for the current digit class
        lowest_values[int(digit_class)] = min_row

    return lowest_values

def plot_mean_distances_x2(training_correct, training_incorrect, testing_correct, testing_incorrect, save=False):
    """
    Plots bar charts of mean distances to centroids for training and testing datasets side by side, with each class represented by two bars.

    Parameters:
    - training_correct: A numpy array containing the mean distances for correct predictions in the training dataset.
    - training_incorrect: A numpy array containing the mean distances for incorrect predictions in the training dataset.
    - testing_correct: A numpy array containing the mean distances for correct predictions in the testing dataset.
    - testing_incorrect: A numpy array containing the mean distances for incorrect predictions in the testing dataset.
    - save: Boolean, if True, saves the plot. Defaults to False.
    """
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Define the clusters and bar width
    clusters = np.arange(len(training_correct))  # Assuming all arrays have the same size
    bar_width = 0.35  # Width of the bars

    # Plot training mean distances
    ax1.bar(clusters - bar_width/2, training_correct, width=bar_width, color='skyblue', label='Correct')
    ax1.bar(clusters + bar_width/2, training_incorrect, width=bar_width, color='orange', label='Incorrect')
    ax1.set_xlabel('Cluster (Digit)')
    ax1.set_ylabel('Mean Distance to Centroid')
    ax1.set_title('Training Data - Mean Distance to Centroid for Each Cluster')
    ax1.set_xticks(clusters)
    ax1.set_xticklabels([str(i) for i in clusters])
    ax1.legend()

    # Plot testing mean distances
    ax2.bar(clusters - bar_width/2, testing_correct, width=bar_width, color='lightgreen', label='Correct')
    ax2.bar(clusters + bar_width/2, testing_incorrect, width=bar_width, color='red', label='Incorrect')
    ax2.set_xlabel('Cluster (Digit)')
    ax2.set_ylabel('Mean Distance to Centroid')
    ax2.set_title('Testing Data - Mean Distance to Centroid for Each Cluster')
    ax2.set_xticks(clusters)
    ax2.set_xticklabels([str(i) for i in clusters])
    ax2.legend()

    plt.tight_layout()

    # Save the plot if required
    if save:
        plt.savefig('Combined_mean_distances.png')

    # Show the plot
    plt.show()

# Example usage with dummy data:
# training_correct = np.random.rand(10)
# training_incorrect = np.random.rand(10)
# testing_correct = np.random.rand(10)
# testing_incorrect = np.random.rand(10)
# plot_mean_distances_x2(training_correct, training_incorrect, testing_correct, testing_incorrect)
