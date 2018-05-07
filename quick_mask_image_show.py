from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time

IMAGE_DIR_PATH = '../images_masks/images/'
MASK_DIR_PATH = '../images_masks/masks/'

# Create list of paths for images and masks
db_imgs = glob(IMAGE_DIR_PATH + '*.png')
db_msks = glob(MASK_DIR_PATH + '*.png')

# Prune pre-augmented images and masks
image_paths = [img for img in db_imgs if 'm90' not in img and 'p90' not in img]
mask_paths = [msk for msk in db_msks if 'm90' not in msk and 'p90' not in msk]

for img, msk in zip(image_paths, mask_paths):
	msk = np.asarray(Image.open(msk))
	img = np.asarray(Image.open(img))
	plt.imshow(img)
	plt.imshow(msk, alpha=.4)
	plt.show()
	plt.close()
	time.sleep(1.5)
