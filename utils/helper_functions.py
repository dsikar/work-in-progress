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


def initialise_data(model_desc, weights_path, commit_hash, repo_url, script_name):
    """
    Initialize the data dictionary with metadata.
    
    Args:
    - model_desc (str): Description or name of the model.
    - weights_path (str): Path to the model's weights file.
    - commit_hash (str): Git commit hash.
    - repo_url (str): URL of the git repository.
    
    Returns:
    - dict: Initialized data dictionary.
    """
    data = {
        'metadata': {
            'model': model_desc,
            'weights_filepath': weights_path,
            'git_commit_hash': commit_hash,
            'git_repository': repo_url,
            'script_name': script_name
        },
        'results': []  # This is an empty list that will hold the results later
    }
    
    return data