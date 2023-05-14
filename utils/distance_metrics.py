import numpy as np

class DistanceMetric:
    def __init__(self, num_channels=3, num_bins=256, val_range=(0, 255), epsilon=1e-10):
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.val_range = val_range
        self.epsilon = epsilon

    def normalized_root_product_sum(self, hist1, hist2):
        """
        Parameters
        -------
            hist1: numpy array
            hist2: numpy array
        Example
        -------
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img1 = mpimg.imread('horses.jpg')
        img2 = mpimg.imread('horses.jpg')
        hist1 = channel_histograms(img1)
        hist2 = channel_histograms(img2)
        normalized_root_product_sum(hist1, hist2)
        """
        intersection_sum = 0
        # Compute the sum of all bin heights
        total_bin_heights = np.sum(hist1)

        for h1, h2 in zip(hist1, hist2):
            
            # Compute the square root of the product of every bin and divide by the sum of all bin heights
            normalized_intersection = np.sum(np.sqrt(np.multiply(h1, h2)) / total_bin_heights) 
            
            intersection_sum += normalized_intersection
            
        return intersection_sum

    def channel_histograms(self, img):
        """
        Compute the histograms for each channel in the image
        Parameters
        -------
            img: uint8 numpy image array
        Example
        -------
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread('horses.jpeg')
        channel_histograms(img)
        """
        histograms = []
        if self.val_range[0] == -1:
            img += 1 # shift values to be in range [0, 2]
     
        # Iterate over the channels in the image
        for channel in range(self.num_channels): # range(img.shape[2]):
            if self.num_channels == 1:
                channel_data = img
            else:   
                channel_data = img[:, :, channel]
            
            # Compute the histogram for the current channel
            hist, _ = np.histogram(channel_data, bins=self.num_bins)
            
            histograms.append(hist) 

        # restore
        if self.val_range[0] == -1:
            img -= 1 # shift values to original range

        return histograms

    def BhattacharyaDistance(self, img1, img2):
        """
        Parameters
        -------
        img1: uint8 numpy image array
        img2: uint8 numpy image array
        Example
        -------
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img1 = mpimg.imread('horses.jpeg')
        img2 = mpimg.imread('horses.jpeg')
        Bhattacharyya_rgb_distance(img1, img2)
        """
        hist1 = self.channel_histograms(img1)
        hist2 = self.channel_histograms(img2)
        nrps = self.normalized_root_product_sum(hist1, hist2) 
        return np.log10(self.epsilon) if nrps == 0 else -np.log10(nrps)

    def HistogramIntersection(self, img1, img2):
        """
        Parameters
        -------
        img1: uint8 numpy image array
        img2: uint8 numpy image array
        Example
        -------
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img1 = mpimg.imread('horses.jpeg')
        img2 = mpimg.imread('horses.jpeg')
        HistogramIntersection(img1, img2)
        """   
        hist1 = self.channel_histograms(img1)
        hist2 = self.channel_histograms(img2)
        intersection_sum = 0
        for h1, h2 in zip(hist1, hist2):
            intersection_sum += np.sum(np.minimum(h1, h2))
        return intersection_sum / np.product(img1.shape)
    
    def KLDivergence(self, img1, img2):
        """
        Parameters
        -------
        img1: uint8 numpy image array
        img2: uint8 numpy image array
        Example
        -------
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img1 = mpimg.imread('horses.jpeg')
        img2 = mpimg.imread('horses.jpeg')
        KLDivergence(img1, img2)
        """
        hist1 = self.channel_histograms(img1)
        hist2 = self.channel_histograms(img2)

        # Compute the normalization constant (sum of all bins in the first histogram)
        normalization_constant = np.product(img1.shape)

        # Compute the log of the height of all bins in the first histogram + epsilon divided by the heights of all bins in the second histogram + epsilon
        log_ratio = 0

        for h1, h2 in zip(hist1, hist2):
            log_ratio += (h1 / normalization_constant) * np.log((h1 + self.epsilon) / (h2 + self.epsilon))

        # Return the sum of the computed values
        return log_ratio.sum()

# helper functions
def get_distances_grayscale(img1, img2, num_channels=1, num_bins=256, val_range=(-1, 1)):
    """
    Compute all distances between two grayscale images
    Parameters
    -------
    img1: uint8 numpy image array
    img2: uint8 numpy image array
    Example
    -------
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img1 = mpimg.imread('horses.jpeg')
    img2 = mpimg.imread('horses.jpeg')
    get_distances_grayscale(img1, img2)
    """
    dm = DistanceMetric(num_channels=num_channels, num_bins=num_bins, val_range=val_range, epsilon=1e-10)
    return {
        'Bhattacharya': dm.BhattacharyaDistance(img1, img2),
        'HistogramIntersection': dm.HistogramIntersection(img1, img2),
        'KLDivergence': dm.KLDivergence(img1, img2)
    }

def get_distances_grayscale_labels(img1, img2, num_channels=1, num_bins=256, val_range=(-1, 1)):
    """
    Compute all distances between two grayscale images
    Parameters
    -------
    img1: uint8 numpy image array
    img2: uint8 numpy image array
    Example
    -------
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img1 = mpimg.imread('horses.jpeg')
    img2 = mpimg.imread('horses.jpeg')
    get_distances_grayscale(img1, img2)
    """
    dm = DistanceMetric(num_channels=1, num_bins=num_bins, val_range=(-1, 1), epsilon=1e-10)
    # return "{:.2f}|{:.2f}|{:.2f}".format(dm.HistogramIntersection(img1, img2), dm.KLDivergence(img1, img2), dm.BhattacharyaDistance(img1, img2))
    return "{:.2f}|{:.2f}".format(dm.KLDivergence(img1, img2), dm.BhattacharyaDistance(img1, img2))
    
    # return {
    #     'Bhattacharya': dm.BhattacharyaDistance(img1, img2),
    #     'HistogramIntersection': dm.HistogramIntersection(img1, img2),
    #     'KLDivergence': dm.KLDivergence(img1, img2)
    # }    

def get_distances_RGB(img1, img2):
    """
    Compute all distances between two RGB images
    Parameters
    -------
    img1: uint8 numpy image array
    img2: uint8 numpy image array
    Example
    -------
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img1 = mpimg.imread('horses.jpeg')
    img2 = mpimg.imread('horses.jpeg')
    get_distances_RGB(img1, img2)
    """
    dm = DistanceMetric(num_channels=3, num_bins=256, val_range=(0, 255), epsilon=1e-10)
    return {
        'Bhattacharya': dm.BhattacharyaDistance(img1, img2),
        'HistogramIntersection': dm.HistogramIntersection(img1, img2),
        'KLDivergence': dm.KLDivergence(img1, img2)
    }

def get_distances_RGB_labels(img1, img2):
    """
    Compute all distances between two RGB images
    Parameters
    -------
    img1: uint8 numpy image array
    img2: uint8 numpy image array
    Example
    -------
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img1 = mpimg.imread('horses.jpeg')
    img2 = mpimg.imread('horses.jpeg')
    get_distances_RGB(img1, img2)
    """
    dm = DistanceMetric(num_channels=3, num_bins=256, val_range=(0, 255), epsilon=1e-10)
    return {
        'Bhattacharya': dm.BhattacharyaDistance(img1, img2),
        'HistogramIntersection': dm.HistogramIntersection(img1, img2),
        'KLDivergence': dm.KLDivergence(img1, img2)
    }        
# import numpy as np
# from scipy.spatial.distance import jensenshannon

# class Distance:

#     def __init__(self):
#         pass

#     def histogram_overlap(self, img1, img2, num_bins=256):
#         img1_min, img1_max = np.min(img1), np.max(img1)
#         img2_min, img2_max = np.min(img2), np.max(img2)

#         bins = np.linspace(min(img1_min, img2_min), max(img1_max, img2_max), num_bins+1)
#         hist1, _ = np.histogram(img1, bins=bins)
#         hist2, _ = np.histogram(img2, bins=bins)

#         overlap = np.minimum(hist1, hist2).sum() / np.maximum(hist1, hist2).sum()
#         return overlap

#     def kl_divergence(self, img1, img2, num_bins=256):
#         img1_min, img1_max = np.min(img1), np.max(img1)
#         img2_min, img2_max = np.min(img2), np.max(img2)

#         bins = np.linspace(min(img1_min, img2_min), max(img1_max, img2_max), num_bins+1)
#         hist1, _ = np.histogram(img1, bins=bins, density=True)
#         hist2, _ = np.histogram(img2, bins=bins, density=True)

#         hist1 += 1e-8
#         hist2 += 1e-8
#         kl_div = np.sum(hist1 * np.log(hist1 / hist2))

#         return kl_div

#     def bhattacharyya_distance(self, img1, img2, num_bins=256):
#         img1_min, img1_max = np.min(img1), np.max(img1)
#         img2_min, img2_max = np.min(img2), np.max(img2)

#         bins = np.linspace(min(img1_min, img2_min), max(img1_max, img2_max), num_bins+1)
#         hist1, _ = np.histogram(img1, bins=bins, density=True)
#         hist2, _ = np.histogram(img2, bins=bins, density=True)

#         b_distance = -np.log(np.sum(np.sqrt(hist1 * hist2)))

#         return b_distance
