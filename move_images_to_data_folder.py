from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import time
import shutil

IMAGE_DIR_PATH = '../VAMdata2/images/'
MASK_DIR_PATH = '../VAMdata2/masks/'

train_targ_dir = '../data_folder/train/'
test_targ_dir = '../data_folder/test/'
train_perc = 0.8

# Create list of paths for images and masks
db_imgs = glob(IMAGE_DIR_PATH + '*.png')
db_msks = glob(MASK_DIR_PATH + '*.png')

# Prune pre-augmented images and masks
image_paths = [img for img in db_imgs if 'm90' not in img and 'p90' not in img]
mask_paths = [msk for msk in db_msks if 'm90' not in msk and 'p90' not in msk]

image_paths.sort()
mask_paths.sort()


for i in range(len(image_paths)):
	if i < len(image_paths)*train_perc:
		shutil.copy(image_paths[i], train_targ_dir+'img')
		shutil.copy(mask_paths[i], train_targ_dir+'gt')

	else:
		shutil.copy(image_paths[i], test_targ_dir+'img')
		shutil.copy(mask_paths[i], test_targ_dir+'gt')

