import matplotlib.pyplot as plt
import numpy as np
import torch

# Functions to help visualise the images, array values, histograms and other useful data

# CIFAR10 specific

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

import numpy as np

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

def show_img(input):
    # show a single image 
    import matplotlib.pyplot as plt

    # select a single image from the inputs tensor
    image = input

    # plot the image
    if image.shape[0] == 1:
        # grayscale image
        plt.imshow(image.squeeze(), cmap='gray')
    else:
        # RGB image
        plt.imshow(image.permute(1, 2, 0))
    plt.show()

def find_max_value(tensor):
    # find the index of the highest value in the tensor
    index = torch.argmax(tensor)

    # get the value at the highest index
    value = tensor[index]

    # return the index and value as a tuple
    return index.item(), value.item()


def initialise_data(model_desc, weights_path, commit_hash, repo_url, script_name, accuracy):
    """
    Initialize the data dictionary with metadata.
    
    Args:
    - model_desc (str): Description or name of the model.
    - weights_path (str): Path to the model's weights file.
    - commit_hash (str): Git commit hash.
    - repo_url (str): URL of the git repository.
    - script_name (str): Name of the script that generated the data.
    - accuracy (float): Model's accuracy on the original test data.
    
    Returns:
    - dict: Initialized data dictionary.
    """
    data = {
        'metadata': {
            'model': model_desc,
            'weights_filepath': weights_path,
            'git_commit_hash': commit_hash,
            'git_repository': repo_url,
            'script_name': script_name,
            'accuracy': accuracy
        },
        'results': []  # This is an empty list that will hold the results later
    }
    
    return data

def append_results(data, index, bd, kl, hi, noise_type, accuracy):
    """
    Append a dictionary of results to the 'results' list in the data.
    
    Args:
    - data (dict): The main data dictionary.
    - index (int): Index representing the "level" of noise.
    - bd (float): BD metric value.
    - kl (float): KL metric value.
    - hi (float): HI metric value.
    - noise_type (str): Type of noise.
    - accuracy (float): Model's accuracy on perturbed data.
    """
    result = {
        'index': index,
        'bd': bd,
        'kl': kl,
        'hi': hi,
        'noise_type': noise_type,
        'accuracy': accuracy
    }
    
    data['results'].append(result)

import pickle
from datetime import datetime

def save_to_pickle(data, filename_prefix):
    """
    Save data to a pickle file with a timestamp suffix.
    
    Args:
    - data (dict): Data to be saved.
    - filename_prefix (str): Prefix for the filename.
    
    Returns:
    - str: The path to the saved file.
    """
    # Generate the timestamp suffix
    timestamp = datetime.now().strftime("_%Y%m%d%H%M%S")
    filename = filename_prefix + timestamp + ".pkl"
    
    # Save data to pickle file
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    
    return filename

def frost_mask_plot(image, frost_image, noisy_image):
    """
    Plot original image, frost mask and Noisy image.
    """

    fig, axs = plt.subplots(1, 6, figsize=(18, 3))

    # Plot original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')

    # Plot histogram of original image
    axs[1].hist(image.flatten(), bins=20)
    axs[1].set_title('Original Image Histogram')

    # Plot frost_image
    axs[2].imshow(frost_image, cmap='gray')
    axs[2].set_title('Frost Image')

    # Plot histogram of frost_image
    axs[3].hist(frost_image.flatten(), bins=20)
    axs[3].set_title('Frost Image Histogram')

    # Plot noisy_image
    axs[4].imshow(noisy_image, cmap='gray')
    axs[4].set_title('Noisy Image')

    # Plot histogram of noisy_image
    axs[5].hist(noisy_image.flatten(), bins=20)
    axs[5].set_title('Noisy Image Histogram')

    plt.tight_layout()
    plt.show()    

