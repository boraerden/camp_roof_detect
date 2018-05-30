from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from pdb import set_trace as t
from time import sleep

msk_dir = '../new_imgs_mks/newmasks/*.png'

new_msk_dir = '../new_imgs_mks/newmasks_width_plus10/'

if not os.path.isdir(new_msk_dir):
	os.makedirs(new_msk_dir)


def widen_mask(msk):
	wider_msk = np.copy(msk)

	width, height = msk.shape
	for i in range(10):
		for x in range(1,width-1):
			for y in range(1,height-1):
				if msk[x,y] > 0:
					wider_msk[x-1,y-1]= 1
					wider_msk[x+1,y-1]= 1
					wider_msk[x-1,y+1]= 1
					wider_msk[x+1,y+1]= 1

		msk = np.copy(wider_msk)

	return wider_msk

msk_paths = glob(msk_dir)

for msk_path in tqdm(msk_paths):
	
	msk = np.asarray(Image.open(msk_path))
	wider_msk_Image = Image.fromarray(widen_mask(msk))

	new_path = new_msk_dir + msk_path.split('/')[-1]
	wider_msk_Image.save(new_path)


