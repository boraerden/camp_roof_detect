from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time

IMAGE_DIR_PATH = '../new_imgs_mks/newimages/'
MASK_DIR_PATH = '../new_imgs_mks/newmasks_width_plus10/'


# Create list of paths for images and masks
image_paths = glob(IMAGE_DIR_PATH + '*.png')
mask_paths = glob(MASK_DIR_PATH + '*.png')

for img, msk in zip(image_paths, mask_paths):
	msk = np.asarray(Image.open(msk))
	img = np.asarray(Image.open(img))
	plt.imshow(img)
	plt.imshow(msk, alpha=.4)
	plt.show()
	plt.close()
	time.sleep(1.5)
