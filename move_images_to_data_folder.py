from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import time
import shutil

IMAGE_DIR_PATH = '../new_imgs_mks/newimages/'
MASK_DIR_PATH = '../new_imgs_mks/newmasks/'

targ_dir = '../new_and_old_images_masks/'

# Create list of paths for images and masks
db_imgs = glob(IMAGE_DIR_PATH + '*.png')
db_msks = glob(MASK_DIR_PATH + '*.png')

# Prune pre-augmented images and masks
image_paths = [img for img in db_imgs if 'm90' not in img and 'p90' not in img]
mask_paths = [msk for msk in db_msks if 'm90' not in msk and 'p90' not in msk]

image_paths.sort()
mask_paths.sort()

for i in range(len(image_paths)):
	shutil.copy(image_paths[i], targ_dir+'images')
	shutil.copy(mask_paths[i], targ_dir+'masks')

