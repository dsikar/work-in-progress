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

# from utils.distance_metrics import DistanceMetric

# # open an image
 
# from PIL import Image
# import numpy as np

# img = Image.open(current_dir + '/images/mnist_nine_grayscale.png')
# img_arr = np.array(img)

# plt.imshow(img_arr, cmap='gray')
# plt.show()


# load the pickle file into memory
with open(current_dir + '/images/original_array.pkl', 'rb') as f:
    original_array = pickle.load(f)
# load the modified pickle file into memory
with open(current_dir + '/images/modified_array_plus_one.pkl', 'rb') as f:
    modified_array = pickle.load(f)

# plot the array
plt.imshow(original_array, cmap='gray')
plt.show()    
