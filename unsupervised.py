import cv2 as cv
from utils import *
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import scipy as sp
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pdb import set_trace as t
from tqdm import tqdm
from random import shuffle


def featurizer_color(img):

	# grab the image channels, initialize the tuple of colors,
	# the figure and the flattened feature vector
	chans = cv.split(img)
	features = []
	 
	# loop over the image channels
	for (chan, color) in zip(chans, ("b", "g", "r")):
		# create a histogram for the current channel and
		# concatenate the resulting histograms for each
		# channel
		hist = cv.calcHist([chan], [0], None, [256], [0, 256])
		features.extend(hist)
		
	return features

def featurizer_hu(img):
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	_,img = cv.threshold(img,127,255,cv.THRESH_TOZERO)
	hu = cv.HuMoments(cv.moments(img)).flatten()
	hu_feat = np.log(np.abs(hu))
	return hu

def img_msk_to_house_chips(img_paths, msk_paths):
	house_chips = []
	for img_path, msk_path in tqdm(zip(img_paths, msk_paths)):

		contours = mask_to_contours(msk_path)
		contours = widen_contours(contours, 20)
		boxes = contours_to_xy_minmaxs(contours)
		img = cv.imread(img_path)
		for xmin, ymin, xmax, ymax in boxes:
			house_chip = img[ymin:ymax,xmin:xmax]
			house_chips.append(house_chip)

	return house_chips


def unsupervised(img_dir, msk_dir):

	print('listing images and masks...')
	img_paths = glob(os.path.join(img_dir,'*.png'))
	msk_paths = glob(os.path.join(msk_dir,'*.png'))

	print('collecting cropped house images...')
	house_chips = img_msk_to_house_chips(img_paths, msk_paths)
	n_house_chips = len(house_chips)

	print('hu featurizing the images...')
	house_feats_hu = []
	for house_chip in tqdm(house_chips):
		house_feats_hu.append(featurizer_hu(house_chip))
	house_feats_hu = np.squeeze(house_feats_hu)

	print('color featurizing the images...')
	house_feats_color = []
	for house_chip in tqdm(house_chips):
		house_feats_color.append(featurizer_color(house_chip))
	house_feats_color = np.squeeze(house_feats_color)

	print('dimensionality reducing color features...')
	pca = PCA(n_components=15)
	house_feats_color_pca = pca.fit_transform(house_feats_color)
	

	house_feats = np.concatenate((house_feats_color_pca,house_feats_hu), axis=1)

	print('kmeans fitting...')
	k = 4
	kmeans = KMeans(n_clusters=k, max_iter=5000)
	clusters = kmeans.fit_predict(house_feats)



	print('showing clusters examples...')
	chip_clusters = []
	for i in range(k):
		cluster_chips = [house_chips[j] for j in range(n_house_chips) if clusters[j]==i]
		chip_clusters.append(cluster_chips)

	n_examples = 20
	fig = plt.figure(figsize=(n_examples, 7))  # width, height in inches

	for i in range(1, k*n_examples + 1):
		c = int(np.floor((i-1)/n_examples))
		fig.add_subplot(k, n_examples, i)
		plt.imshow(chip_clusters[c][i%n_examples])
		plt.axis('off')

	plt.show()



img_dir = '../new_and_old_images_masks/images/'
msk_dir = '../new_and_old_images_masks/masks/'

unsupervised(img_dir, msk_dir)


