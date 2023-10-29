import matplotlib.pyplot as plt
import numpy as np
import torch

# Functions to help visualise the images, array values, histograms and other useful data

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

