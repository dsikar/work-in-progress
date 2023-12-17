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
        {'frost_level': 0.1, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.2, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.3, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.4, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.5, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.6, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.7, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.8, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 0.9, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5},
        {'frost_level': 1.0, 'frost_sigma': 0.5, 'frost_threshold': 0.5, 'blur_kernel_size': 3, 'blur_sigma': 0.5}
    ],
    'gaussian_noise': [
        {'mean': 0.05, 'std': 0.25},
        {'mean': 0.1, 'std': 0.25},
        {'mean': 0.15, 'std': 0.25},
        {'mean': 0.2, 'std': 0.25},
        {'mean': 0.25, 'std': 0.25},
        {'mean': 0.3, 'std': 0.25},
        {'mean': 0.35, 'std': 0.25},
        {'mean': 0.4, 'std': 0.25},
        {'mean': 0.45, 'std': 0.25},
        {'mean': 0.5, 'std': 0.25}
    ],
    'impulse_noise': [
        {'density': 0.25, 'intensity': 1},
        {'density': 0.3, 'intensity': 1},
        {'density': 0.35, 'intensity': 1},
        {'density': 0.4, 'intensity': 1},
        {'density': 0.45, 'intensity': 1},
        {'density': 0.5, 'intensity': 1},
        {'density': 0.55, 'intensity': 1},
        {'density': 0.6, 'intensity': 1},
        {'density': 0.65, 'intensity': 1},
        {'density': 0.7, 'intensity': 1},
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
            {'factor': 0.7},
            {'factor': 0.81},
            {'factor': 0.84},
            {'factor': 1.1},
            {'factor': 1.2},
            {'factor': 1.3},
            {'factor': 1.4},
            {'factor': 1.41},
            {'factor': 1.5},
            {'factor': 1.6}
    ],
    'shot_noise': [
        {'intensity': 0.1},
        {'intensity': 0.2},
        {'intensity': 0.3},
        {'intensity': 0.4},
        {'intensity': 0.5},
        {'intensity': 0.6},
        {'intensity': 0.7},
        {'intensity': 0.8},
        {'intensity': 0.9},
        {'intensity': 1.0}
    ],
    'snow': [
            {'snow_level': 0.81, 'snow_color': 1.0, 'blur_kernel_size': 5, 'blur_sigma': 1.0},
            {'snow_level': 0.82, 'snow_color': 1.0, 'blur_kernel_size': 5, 'blur_sigma': 1.0},
            {'snow_level': 0.83, 'snow_color': 1.0, 'blur_kernel_size': 4, 'blur_sigma': 1.0},
            {'snow_level': 0.84, 'snow_color': 1.0, 'blur_kernel_size': 3, 'blur_sigma': 1.0},
            {'snow_level': 0.85, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0},
            {'snow_level': 0.86, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0},
            {'snow_level': 0.87, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0},
            {'snow_level': 0.88, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0},
            {'snow_level': 0.89, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0},
            {'snow_level': 0.9, 'snow_color': 1.0, 'blur_kernel_size': 2, 'blur_sigma': 1.0}
        ],
    'zoom_blur': [
        {'kernel_size': 1, 'strength': 1.0},
        {'kernel_size': 2, 'strength': 1.0},
        {'kernel_size': 3, 'strength': 1.0},
        {'kernel_size': 4, 'strength': 1.0},
        {'kernel_size': 5, 'strength': 1.0},
        {'kernel_size': 6, 'strength': 1.0},
        {'kernel_size': 7, 'strength': 1.0},
        {'kernel_size': 8, 'strength': 1.0},
        {'kernel_size': 9, 'strength': 1.0},
        {'kernel_size': 10, 'strength': 1.0}
    ]
}
