import numpy as np
import cv2

class Perturbation:
    """
    A class to apply various types of noise to grayscale images.

    Args:
        pixel_range (tuple): The range of pixel values in the image (default: (-1, 1)).
    """

    def __init__(self, pixel_range=(-1, 1)):
        self.pixel_range = pixel_range

    def add_brightness(self, image, brightness = 0.1):
        """
        Add brightness to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            brightness_level (float): The amount of brightness to add (default: 0.1).

        Returns:
            ndarray: The brighter image.
        """
        noisy_image = np.clip(image + brightness, *self.pixel_range)
        return noisy_image  
    
    def add_contrast(self, image, contrast_level=0.1):
        """
        Add contrast to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            contrast_level (float): The amount of contrast to add (default: 0.1).

        Returns:
            ndarray: The image with added contrast.
        """
        mean_pixel = np.mean(image)
        noisy_image = np.clip((image - mean_pixel) * (1 + contrast_level), *self.pixel_range)
        return noisy_image    

    def add_defocus_blur(self, image, kernel_size=3, blur_amount=1.0):
        """
        Add defocus blur to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            kernel_size (int): The size of the defocus blur kernel (default: 3).
            blur_amount (float): The amount of defocus blur to apply (default: 1.0).

        Returns:
            ndarray: The blurred image.
        """
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size ** 2
        blurred_image = cv2.filter2D(image, -1, kernel)
        blurred_image = blurred_image * blur_amount + image * (1 - blur_amount)
        blurred_image = np.clip(blurred_image, *self.pixel_range)
        return blurred_image     

    def add_fog(self, image, fog_level=0.1, fog_density=0.5):
        """
        Add fog to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            fog_level (float): The amount of fog to add (default: 0.1).
            fog_density (float): The density of the fog (default: 0.5).

        Returns:
            ndarray: The foggy image.
        """
        height, width = image.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - width/2) / (width/2)
        y = (y - height/2) / (height/2)
        dist = np.sqrt(x**2 + y**2)
        fog_mask = np.exp(-((dist/fog_density)**2))
        fog_mask = np.clip(fog_mask + np.random.normal(scale=0.1, size=(height, width)), 0, 1)
        noisy_image = np.clip(image * (1 - fog_level) + fog_mask * fog_level, *self.pixel_range)
        return noisy_image 

    def add_frost(self, image, frost_level=0.1, frost_sigma=1.0, frost_threshold=0.1, blur_kernel_size=5, blur_sigma=1.0):
        """
        Add frost to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            frost_level (float): The percentage of pixels to turn to frost (default: 0.1).
            frost_sigma (float): The standard deviation of the Gaussian distribution used to generate the frost patterns (default: 1.0).
            frost_threshold (float): The threshold for the frost patterns (default: 0.1).
            blur_kernel_size (int): The size of the Gaussian blur kernel (default: 5).
            blur_sigma (float): The standard deviation of the Gaussian blur (default: 1.0).

        Returns:
            ndarray: The frost-covered image.
        """
        height, width = image.shape
        frost_mask = np.random.normal(loc=0.5, scale=frost_sigma, size=(height, width))
        frost_mask[frost_mask < frost_threshold] = 0
        frost_mask[frost_mask >= frost_threshold] = 1
        frost_mask = np.clip(frost_mask + np.random.normal(scale=0.1, size=(height, width)), 0, 1)
        frost_mask = cv2.GaussianBlur(frost_mask, (blur_kernel_size, blur_kernel_size), blur_sigma)
        frost_image = np.where(frost_mask == 1, np.random.normal(scale=0.1, size=(height, width)), 0)
        noisy_image = np.clip(image + frost_level * frost_image, *self.pixel_range)
        return noisy_image    

    def add_frosted_glass_blur(self, image, kernel_size=3, sigma=1.0):
        """
        Add frosted glass blur to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            kernel_size (int): The size of the frosted glass blur kernel (default: 3).
            sigma (float): The standard deviation of the Gaussian distribution used to add noise to the kernel (default: 1.0).

        Returns:
            ndarray: The blurred image.
        """
        height, width = image.shape
        dx = cv2.randu(np.zeros((height, width), dtype=np.float32), -kernel_size, kernel_size)
        dy = cv2.randu(np.zeros((height, width), dtype=np.float32), -kernel_size, kernel_size)
        noise = np.zeros_like(image)
        cv2.randn(noise, 0, sigma)
        x_indices = np.tile(np.arange(width), (height, 1)) + dx
        y_indices = np.tile(np.arange(height).reshape(-1, 1), (1, width)) + dy
        indices = np.round(np.stack([y_indices, x_indices], axis=-1)).astype(np.int32)
        indices[:, :, 0] = np.clip(indices[:, :, 0], 0, height - 1)
        indices[:, :, 1] = np.clip(indices[:, :, 1], 0, width - 1)
        indices = indices.reshape(-1, 2)
        noisy_kernel = noise[indices[:, 0], indices[:, 1]].reshape(height, width)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size ** 2
        blurred_image = cv2.filter2D(image, -1, kernel + noisy_kernel)
        blurred_image = np.clip(blurred_image, *self.pixel_range)
        return blurred_image 
        
    def add_gaussian_noise(self, image, mean=0, std=1):
        """
        Add Gaussian noise to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            mean (float): The mean of the Gaussian distribution (default: 0).
            std (float): The standard deviation of the Gaussian distribution (default: 1).

        Returns:
            ndarray: The noisy image.
        """
        h, w = image.shape
        noise = np.random.normal(mean, std, (h, w))
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, *self.pixel_range)
        return noisy_image
    
    def add_impulse_noise(self, image, density=0.1, intensity=1):
        """
        Add impulse noise to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            density (float): The density of the impulse noise (default: 0.1).
            intensity (int): The intensity of the impulse noise (default: 1).

        Returns:
            ndarray: The noisy image.
        """
        h, w = image.shape
        noise = np.zeros((h, w), dtype=np.uint8)
        num_pixels = int(density * h * w)
        indices = np.random.choice(h * w, num_pixels, replace=False)
        noise.flat[indices] = intensity
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, *self.pixel_range)
        return noisy_image
    
    def add_motion_blur(self, image, kernel_size=3, angle=0, direction=(1, 0)):
        """
        Add motion blur to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            kernel_size (int): The size of the motion blur kernel (default: 3).
            angle (float): The angle of the motion blur in degrees (default: 0).
            direction (tuple): The direction of the motion blur (default: (1, 0)).

        Returns:
            ndarray: The blurred image.
        """
        height, width = image.shape
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size, dtype=np.float32)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1.0), (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        dx, dy = direction
        if abs(dx) > abs(dy):
            kernel = np.rot90(kernel)
        blurred_image = cv2.filter2D(image, -1, kernel)
        blurred_image = np.clip(blurred_image, *self.pixel_range)
        return blurred_image  

    def add_pixelation(self, image, factor=4):
        """
        Pixelate a grayscale image by replacing each pixel with the average of a square block of pixels.

        Args:
            image (ndarray): The grayscale image.
            factor (int): The factor by which to reduce the size of the image (default: 4).

        Returns:
            ndarray: The pixelated image.
        """
        h, w = image.shape
        h_new, w_new = h // factor, w // factor
        image_small = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)
        image_large = cv2.resize(image_small, (w, h), interpolation=cv2.INTER_NEAREST)
        pixelated_image = ((image_large - image_large.min()) / (image_large.max() - image_large.min())) * (self.pixel_range[1] - self.pixel_range[0]) + self.pixel_range[0]
        return pixelated_image 
       
    def add_shot_noise(self, image, intensity=0.1):
        """
        Add shot noise to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            intensity (float): The intensity of the shot noise (default: 0.1).

        Returns:
            ndarray: The noisy image.
        """
        h, w = image.shape
        noise = np.random.poisson(intensity, (h, w))
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, *self.pixel_range)
        return noisy_image

    def add_snow(self, image, snow_level=0.1, snow_color=1, blur_kernel_size=5, blur_sigma=1.0):
        """
        Add random snow to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            snow_level (float): The percentage of pixels to turn to snow (default: 0.1).
            snow_color (float): The color of the snow (default: 1).
            blur_kernel_size (int): The size of the Gaussian blur kernel (default: 5).
            blur_sigma (float): The standard deviation of the Gaussian blur (default: 1.0).

        Returns:
            ndarray: The snow-covered image.
        """
        height, width = image.shape
        snow_mask = np.zeros((height, width))
        snow_mask[np.random.random((height, width)) < snow_level] = 1
        snow_mask = cv2.GaussianBlur(snow_mask, (blur_kernel_size, blur_kernel_size), blur_sigma)
        snow_mask = np.clip(snow_mask, 0, 1)
        snow_image = np.zeros_like(image)
        snow_image = np.where(snow_mask == 1, snow_color, snow_image)
        noisy_image = np.clip(image + snow_image, *self.pixel_range)
        return noisy_image   
    
    def add_zoom_blur(self, image, kernel_size=3, strength=1.0):
        """
        Add zoom blur to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            kernel_size (int): The size of the zoom blur kernel (default: 3).
            strength (float): The strength of the zoom blur (default: 1.0).

        Returns:
            ndarray: The blurred image.
        """
        height, width = image.shape
        cx, cy = width / 2, height / 2
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx = (j - cx) / (kernel_size / 2)
                dy = (i - cy) / (kernel_size / 2)
                d = np.sqrt(dx ** 2 + dy ** 2)
                r = d * strength
                theta = np.arctan2(dy, dx)
                kernel[i, j] = r * np.sin(theta)
        kernel = kernel / kernel.sum()
        blurred_image = cv2.filter2D(image, -1, kernel)
        blurred_image = np.clip(blurred_image, *self.pixel_range)
        return blurred_image        

    def add_elastic2(self, image, alpha=15, sigma=3, seed=None):
        """
        Apply an elastic transformation to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            alpha (float): The scale factor for the displacement field (default: 15).
            sigma (float): The standard deviation of the Gaussian kernel used to smooth the displacement field (default: 3).
            seed (int): The random seed to use (default: None).

        Returns:
            ndarray: The image with added elastic transformation.
        """
        if seed is not None:
            np.random.seed(seed)
        shape = image.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        noisy_image = cv2.remap(image, np.float32(indices[1]), np.float32(indices[0]), interpolation=cv2.INTER_LINEAR)
        return noisy_image  
        
    def add_elastic(self, image, alpha=200, sigma=10, seed=None):
        """
        Add elastic transformations to a grayscale image.

        Args:
            image (ndarray): The grayscale image.
            alpha (float): The displacement field scale (default: 200).
            sigma (float): The displacement field smoothness (default: 10).
            seed (int): The random seed to use (default: None).

        Returns:
            ndarray: The image with elastic transformations applied.
        """
        if seed is not None:
            np.random.seed(seed)
        image_rgb = np.stack([image]*3, axis=-1)  # Convert grayscale to RGB
        h, w, _ = image_rgb.shape
        dx = np.float32(cv2.getGaussianKernel(w, sigma))
        dy = np.float32(cv2.getGaussianKernel(h, sigma))
        dx = np.outer(dx, np.ones(h))
        dy = np.outer(np.ones(w), dy)
        displacement_field = np.stack([dx, dy], axis=-1)
        displacement_field *= alpha * np.random.randn(h, w, 2)
        displacement_field -= displacement_field.min()
        displacement_field /= displacement_field.max()
        warped_image = cv2.remap(image_rgb, displacement_field[..., 1], displacement_field[..., 0], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_image = ((warped_image - warped_image.min()) / (warped_image.max() - warped_image.min())) * (self.pixel_range[1] - self.pixel_range[0]) + self.pixel_range[0]
        return warped_image[..., 0]  # Convert back to grayscale                   

    def add_jpeg_noise(self, image, quality=80):
        """
        Add JPEG compression artifacts to an image.

        Args:
            image (ndarray): The grayscale image.
            quality (int): The JPEG compression quality (default: 80).

        Returns:
            ndarray: The image with JPEG compression artifacts added.
        """
        # Convert image to JPEG and back to simulate compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", image, encode_param)
        jpeg_image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
        jpeg_image = ((jpeg_image - jpeg_image.min()) / (jpeg_image.max() - jpeg_image.min())) * (self.pixel_range[1] - self.pixel_range[0]) + self.pixel_range[0]
        return jpeg_image 

       