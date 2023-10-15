import sys
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to the module search path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.distance_metrics import DistanceMetric
from utils.perturbations import *
from utils.helper_functions import *
from utils.perturbation_levels import PERTURBATION_LEVELS

# Evaluate the model on the test dataset for different values of noise
pt = Perturbation()
# Note, (-1, 1) is the range of values for the MNIST dataset
dm = DistanceMetric(num_channels=1, num_bins=30, val_range=(-1,1))

# show image
from PIL import Image
import numpy as np
#img = Image.open(current_dir + '/images/mnist_nine_grayscale.png')
#img_arr = np.array(img)
#plt.imshow(img_arr, cmap='gray')
#plt.show()

# with open(current_dir + '/images/original_array.pkl', 'rb') as f:
#     original_array = pickle.load(f)
# # plt.imshow(original_array, cmap='gray')
# # plt.show()

import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MaxNLocator

with open(current_dir + '/images/original_array.pkl', 'rb') as f:
    original_array = pickle.load(f)

# the histogram of the data
n, bins, patches = plt.hist(original_array.flatten(), 30, density=True, facecolor='g', alpha=0.75)
n = n / (original_array.shape[0] * original_array.shape[1]) # normalize the histogram values
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True)) # set y axis values as integers
#plot the histogram
plt.xlabel('Pixel Value')
plt.ylabel('Probability')
plt.title('Histogram of Pixel Values')
plt.grid(True)
plt.show()


fig, axs = plt.subplots(1, 11, figsize=(20, 20))
#for i in range(11):
axs[0].imshow(original_array, cmap='gray')
axs[0].axis('off')
axs[0].set_title("0")  # Add label "0" to every image

bd = dm.BhattacharyaDistance(original_array, original_array)
kl = dm.KLDivergence(original_array, original_array)
hi = dm.HistogramIntersection(original_array, original_array)

# Add bd, kl, and hi below each image
axs[0].text(0.5, -0.1, f'BD: {bd:.2f}', ha='center', va='center', transform=axs[0].transAxes)
axs[0].text(0.5, -0.2, f'KL: {kl:.2f}', ha='center', va='center', transform=axs[0].transAxes)
axs[0].text(0.5, -0.3, f'HI: {hi:.2f}', ha='center', va='center', transform=axs[0].transAxes)    

perturbed_arrays= []

key = 'brightness'
for k in range(0, len(PERTURBATION_LEVELS[key])):
    kwargs = PERTURBATION_LEVELS[key][k]
    print(kwargs)
    perturbed_array = getattr(pt, key)(original_array, **kwargs)
    # save the perturbed array to a list
    perturbed_arrays.append(perturbed_array)

    bd = dm.BhattacharyaDistance(original_array, perturbed_array)
    kl = dm.KLDivergence(original_array, perturbed_array)
    hi = dm.HistogramIntersection(original_array, perturbed_array)
    axs[k+1].imshow(perturbed_arrays[k], cmap='gray')
    axs[k+1].axis('off')
    axs[k+1].set_title(k+1)    
    
    # Add bd, kl, and hi below each image
    axs[k+1].text(0.5, -0.1, f'BD: {bd:.5f}', ha='center', va='center', transform=axs[k+1].transAxes)
    axs[k+1].text(0.5, -0.2, f'KL: {kl:.5f}', ha='center', va='center', transform=axs[k+1].transAxes)
    axs[k+1].text(0.5, -0.3, f'HI: {hi:.5f}', ha='center', va='center', transform=axs[k+1].transAxes)    

plt.tight_layout()  # Reduce spacing and adjust borders    
plt.show()
# save the plot
fig.savefig(current_dir + '/images/brightness_perturbations_30_bins.png', bbox_inches='tight')

