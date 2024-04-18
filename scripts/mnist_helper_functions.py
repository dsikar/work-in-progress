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

def single_plot_accuracy_decrements(results, results_overall, save=True):
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
        ax.plot(decrement, accuracy, marker='o', color=cmap(i), label=f'Digit {int(digit_class)}')

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
    ax.set_title('Class Prediction Accuracy vs Distance to Threshold Decrement')

    # Add a legend on the bottom left
    ax.legend(loc='lower left', ncol=2)

    # Adjust the spacing
    plt.tight_layout()

    # Save the plot as an image file
    if save:
        plt.savefig('single_plot_accuracy_decrements.png')

    # Display the plot
    plt.show()