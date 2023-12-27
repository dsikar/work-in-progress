PERTURBATION_LEVELS = {
    'brightness': [
        {'brightness_level': 0.1},
        {'brightness_level': 0.55},
        {'brightness_level': 0.65},
        {'brightness_level': 0.7},
        {'brightness_level': 0.75},
        {'brightness_level': 0.8},
        {'brightness_level': 0.85},
        {'brightness_level': 0.9},
        {'brightness_level': 0.95},
        {'brightness_level': 1.0},
    ],
    'contrast': [
        {'contrast_level': 3.0},
        {'contrast_level': 2.1},
        {'contrast_level': 1.7},
        {'contrast_level': 1.4},
        {'contrast_level': 1.0},
        {'contrast_level': 0.2},
        {'contrast_level': -0.2},
        {'contrast_level': -0.4},
        {'contrast_level': -0.5},
        {'contrast_level': -0.6},
    ],
    'defocus_blur': [
        {'kernel_size': 3, 'blur_amount': 0.1}, # * 98
        {'kernel_size': 7, 'blur_amount': 0.6}, # * 94        
        {'kernel_size': 7, 'blur_amount': 0.7}, # * 90         
        {'kernel_size': 6, 'blur_amount': 0.8}, # * 85 
        {'kernel_size': 6, 'blur_amount': 1.5}, # * 80        
        {'kernel_size': 6, 'blur_amount': 0.9}, # * 76 
        {'kernel_size': 6, 'blur_amount': 1.0}, # * 69      
        {'kernel_size': 7, 'blur_amount': 1.2}, # * 65
        {'kernel_size': 7, 'blur_amount': 1.0}, # * 57     
        {'kernel_size': 8, 'blur_amount': 1.2}, # * 51  
    ],
    'fog': [
        {'fog_level': 0.1, 'fog_density': 0.1}, # 98.5
        {'fog_level': 0.43, 'fog_density': 0.43}, # * 94
        {'fog_level': 0.45, 'fog_density': 0.45}, # * 92
        {'fog_level': 0.48, 'fog_density': 0.48}, # * 87
        {'fog_level': 0.5, 'fog_density': 0.5}, # * 81
        {'fog_level': 0.52, 'fog_density': 0.52}, # * 76
        {'fog_level': 0.54, 'fog_density': 0.54}, # * 71
        {'fog_level': 0.56, 'fog_density': 0.56}, # * 65
        {'fog_level': 0.58, 'fog_density': 0.58}, # * 58
        {'fog_level': 0.6, 'fog_density': 0.6}, # * 50
    ],
    'frost': [
        {'frost_level': 6, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 97 *
        {'frost_level': 8, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 93 *
        {'frost_level': 10, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 85 *
        {'frost_level': 11, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 80 *
        {'frost_level': 12, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 75 *
        {'frost_level': 13, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 69 *
        {'frost_level': 14, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 65 *
        {'frost_level': 16, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 58 *
        {'frost_level': 17, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 54 *
        {'frost_level': 18, 'frost_sigma': 1.0, 'frost_threshold': 0.1, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 50 *
    ],
    'gaussian_noise': [
        {'mean': 0.05, 'std': 0.25},  #98.31
        {'mean': 0.45, 'std': 0.25}, # 93.78
        {'mean': 0.55, 'std': 0.25}, # 85.74
        {'mean': 0.6, 'std': 0.25}, # 80
        {'mean': 0.65, 'std': 0.25}, # 76.1
        {'mean': 0.7, 'std': 0.25}, # 71
        {'mean': 0.75, 'std': 0.25},  # 66.1
        {'mean': 0.8, 'std': 0.25}, # 61
        {'mean': 0.835, 'std': 0.25}, # 55.3
        {'mean': 0.87, 'std': 0.25}, # 50
    ],
    'impulse_noise': [
        {'density': 0.25, 'intensity': 1},
        {'density': 0.35, 'intensity': 1},
        {'density': 0.4, 'intensity': 1},
        {'density': 0.45, 'intensity': 1},
        {'density': 0.5, 'intensity': 1},
        {'density': 0.55, 'intensity': 1},
        {'density': 0.6, 'intensity': 1},
        {'density': 0.65, 'intensity': 1},
        {'density': 0.7, 'intensity': 1},
        {'density': 0.75, 'intensity': 1},
    ],
    'motion_blur': [
        {'kernel_size': 1, 'angle': 0.0, 'direction': (1.0, 0.0)},
        {'kernel_size': 2, 'angle': 0.0, 'direction': (2.0, 0.0)},
        {'kernel_size': 3, 'angle': 0.0, 'direction': (3.0, 0.0)},
        {'kernel_size': 4, 'angle': 0.0, 'direction': (4.0, 0.0)},
        {'kernel_size': 5, 'angle': 0.0, 'direction': (5.0, 0.0)},
        {'kernel_size': 6, 'angle': 0.0, 'direction': (6.0, 0.0)},
        {'kernel_size': 7, 'angle': 0.0, 'direction': (7.0, 0.0)},
        {'kernel_size': 8, 'angle': 0.0, 'direction': (8.0, 0.0)},
        {'kernel_size': 9, 'angle': 0.0, 'direction': (9.0, 0.0)},
        {'kernel_size': 10, 'angle': 0.0, 'direction': (10.0, 0.0)}
    ],
    'pixelation': [
        {'factor': 0.7}, # 97.15 *
        {'factor': 2}, # 98.09 *
        {'factor': 2.5}, # 95.0 *
        {'factor': 3}, # 93.84 *
        {'factor': 3.5}, # 89.76 *
        {'factor': 3.55}, # 84.01 *
        {'factor': 3.75}, # 84.01 *
        {'factor': 4.25}, # 72.800 * 
        {'factor': 4.7}, # 55.5000 *
        {'factor': 5.7}, # 43.21 *
    ],
    'shot_noise': [
        {'intensity': 0.1}, # 97
        {'intensity': 0.2}, # 95
        {'intensity': 0.25}, # 79
        {'intensity': 0.3}, #87
        {'intensity': 0.35}, # 80
        {'intensity': 0.4}, # 74
        {'intensity': 0.42}, # 71
        {'intensity': 0.46}, # 66
        {'intensity': 0.5}, # 60
        {'intensity': 0.57}, # 51
    ],
    'snow': [
        {'snow_level': 0.81, 'snow_color': 1.0, 'blur_kernel_size': 5, 'blur_sigma': 1.0}, # 98 *
        {'snow_level': 0.84, 'snow_color': 1.0, 'blur_kernel_size': 3, 'blur_sigma': 1.0}, # 92 *
        {'snow_level': 0.86, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 89 *
        {'snow_level': 0.88, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 85 *
        {'snow_level': 0.9, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 83 *
        {'snow_level': 0.91, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 80 *
        {'snow_level': 0.93, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 73 *
        {'snow_level': 0.94, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 69 *
        {'snow_level': 0.95, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 66 *
        {'snow_level': 1.0, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}, # 49 *
        ],
    'zoom_blur': [
        {'kernel_size': 1, 'strength': 1.0}, # 98
        {'kernel_size': 2, 'strength': 1.0}, # 96
        {'kernel_size': 3, 'strength': 1.0}, # 96 
        {'kernel_size': 4, 'strength': 1.0}, # 89
        {'kernel_size': 5, 'strength': 1.0}, # 85
        {'kernel_size': 5, 'strength': 1.0}, # 85 repeat
        {'kernel_size': 6, 'strength': 1.0}, # 67
        {'kernel_size': 6, 'strength': 1.0}, # 67 repeat
        {'kernel_size': 7, 'strength': 1.0}, # 58
        {'kernel_size': 8, 'strength': 1.0}, # 36
    ]
}
