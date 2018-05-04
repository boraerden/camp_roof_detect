from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

aug_imgs = glob('../../VAMdata2/images/CAR*.png')
imgs = [img for img in aug_imgs if 'm90' not in img and 'p90' not in img]
aug_msks = glob('../../VAMdata2/masks/CAR*.png')
msks = [msk for msk in aug_msks if 'm90' not in msk and 'p90' not in msk]

for img, msk in zip(imgs, msks):
	msk = np.asarray(Image.open(msk))
	img = np.asarray(Image.open(img))
	plt.imshow(img)
	plt.imshow(msk, alpha=.4)
	plt.show()
	plt.close()