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


