import numpy as np
import cv2
import matplotlib.pyplot as plt

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
        return intersection_sum / np.prod(img1.shape)
    
    def debugGSHistogramIntersection(self, img1, img2):
        # debug GrayScale Histogram Intersection
        img1 += 1 # shift values to be in range [0, 2]
        img2 += 1
        hist1 = cv2.calcHist([img1], [0], None, [self.num_bins], [0,2])
        hist2 = cv2.calcHist([img2], [0], None, [self.num_bins], [0,2])
        min_hist = np.minimum(hist1, hist2)
        intersection = np.sum(min_hist) / np.prod(img1.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img1, cmap='gray')
        ax1.set_title('Image 1')
        ax2.imshow(img2, cmap='gray')
        ax2.set_title('Image 2')
        fig.suptitle(f'Histogram Intersection (bins={self.num_bins}, overlap={intersection:.2f})')
        fig, ax = plt.subplots()
        ax.plot(hist1, label='Image 1')
        ax.plot(hist2, label='Image 2')
        ax.set_title(f'Channel Histograms (bins={self.num_bins})')
        ax.legend()
        plt.savefig('debug.png')

        return intersection    
    
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
        #normalization_constant = np.product(img1.shape)
        normalization_constant = np.prod(img1.shape)

        # Compute the log of the height of all bins in the first histogram + epsilon divided by the heights of all bins in the second histogram + epsilon
        log_ratio = 0

        for h1, h2 in zip(hist1, hist2):
            log_ratio += (h1 / normalization_constant) * np.log((h1 + self.epsilon) / (h2 + self.epsilon))

        # Return the sum of the computed values
    #    return log_ratio.sum()
        return np.sum(log_ratio)

class DistanceCalculator:
    """
    dm = DistanceMetric(num_channels=3, num_bins=256, val_range=(0, 255), epsilon=1e-10)
    dc = DistanceCalculator(dm)

    img1 = mpimg.imread('horses.jpeg')
    img2 = mpimg.imread('horses.jpeg')

    distances = dc.get_distances_RGB(img1, img2)
    print(distances)
    """
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric
    
    def get_distances_grayscale_labels(self, img1, img2):
        return "{:.2f}|{:.2f}".format(self.distance_metric.KLDivergence(img1, img2), self.distance_metric.BhattacharyaDistance(img1, img2))
    
    def get_distances_RGB(self, img1, img2):
        return {
            'Bhattacharya': self.distance_metric.BhattacharyaDistance(img1, img2),
            'HistogramIntersection': self.distance_metric.HistogramIntersection(img1, img2),
            'KLDivergence': self.distance_metric.KLDivergence(img1, img2)
        }
    
    def get_distances_RGB_labels(self, img1, img2):
        return "{:.2f}|{:.2f}".format(self.distance_metric.KLDivergence(img1, img2), self.distance_metric.BhattacharyaDistance(img1, img2))
    
    def get_distances_grayscale(self, img1, img2, num_channels=1, num_bins=256, val_range=(-1, 1)):
        dm = DistanceMetric(num_channels=num_channels, num_bins=num_bins, val_range=val_range, epsilon=1e-10)
        return {
            'Bhattacharya': self.distance_metric.BhattacharyaDistance(img1, img2),
            'HistogramIntersection': self.distance_metric.HistogramIntersection(img1, img2),
            'KLDivergence': self.distance_metric.KLDivergence(img1, img2)
        }
   
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
    return "{:.2f}|{:.2f}".format(dm.KLDivergence(img1, img2), dm.BhattacharyaDistance(img1, img2))
    # return {
    #     'Bhattacharya': dm.BhattacharyaDistance(img1, img2),
    #     'HistogramIntersection': dm.HistogramIntersection(img1, img2),
    #     'KLDivergence': dm.KLDivergence(img1, img2)
    # }        