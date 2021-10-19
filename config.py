import numpy as np
from easydict import EasyDict as edict

config = edict()

config.enable_blur = True
config.enable_gaussian_noise = True
config.enable_color_jitter = True
config.hand_random_flip = True
config.enable_black_border = True
config.min_rot_angle = -30
config.max_rot_angle = 30

