import numpy as np

def histogram_overlap(image1, image2, num_bins=21):
    """
    Compute the histogram overlap between two image arrays with the shape [1, 28, 28].
    
    Args:
    image1 (np.ndarray): First image array.
    image2 (np.ndarray): Second image array.
    num_bins (int): Number of bins for the histograms.

    Returns:
    float: Normalised histogram overlap between the input image arrays.
    """
    # Shift the values from [-1, 1] to [0 , 2 ]
    shifted_image1 = image1 + 1
    shifted_image2 = image2 + 1

    # Compute the histograms
    hist1, _ = np.histogram(shifted_image1, bins=num_bins, range=(0, 2))
    hist2, _ = np.histogram(shifted_image2, bins=num_bins, range=(0, 2))

    # Normalize the histograms
    #hist1 = hist1 / np.sum(hist1)
    #hist2 = hist2 / np.sum(hist2)

    # Compute the histogram overlap and normalise
    overlap = np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)
    overlap = 0 if overlap is None else overlap
    return overlap
