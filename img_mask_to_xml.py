from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
from pprint import pprint
from pdb import set_trace as t
import xml.etree.cElementTree as ET
from tqdm import tqdm
import argparse
import os
import cv2 as cv
from utils import *


def write_xml(msk_path, boxes, xml_dir):
	write_path = xml_dir + msk_path.split('/')[-1][:-4] + '.xml'


	annotation = ET.Element("annotation")
	folder = ET.SubElement(annotation, "folder").text = '../images/'
	filename = ET.SubElement(annotation, "filename").text = msk_path.split('/')[-1]

	size = ET.SubElement(annotation, "size")
	width = ET.SubElement(size, "width").text = str(256)
	height = ET.SubElement(size, "height").text = str(256)
	depth = ET.SubElement(size, "depth").text = str(3)

	segmented = ET.SubElement(annotation, "segmented").text = str(0)

	for xmin, ymin, xmax, ymax in boxes:
		object_ = ET.SubElement(annotation, "object")
		name = ET.SubElement(object_, "name").text = 'house'
		pose = ET.SubElement(object_, "pose").text = 'Unspecified'
		trun = ET.SubElement(object_, "truncated").text = str(0)
		diff = ET.SubElement(object_, "difficult").text = str(0)

		bndbox = ET.SubElement(object_, "bndbox")
		xmin = ET.SubElement(bndbox, "xmin").text = str(xmin)
		ymin = ET.SubElement(bndbox, "ymin").text = str(ymin)
		xmax = ET.SubElement(bndbox, "xmax").text = str(xmax)
		ymax = ET.SubElement(bndbox, "ymax").text = str(ymax)


	tree = ET.ElementTree(annotation)

	tree.write(write_path)



def main(img_dir, msk_dir, xml_dir):

	# Create list of paths for images and masks
	img_paths = glob(os.path.join(img_dir,'*.png'))
	msk_paths = glob(os.path.join(msk_dir,'*.png'))

	for img_path, msk_path in tqdm(zip(img_paths, msk_paths)):


		contours = mask_to_contours(msk_path)
		contours = widen_contours(contours, 7)

		# img = cv.imread(img_path)
		# cv.drawContours(img, contours, -1,(0,1,0), 3)
		# cv.imshow('title',img)
		# cv.waitKey()

		boxes = contours_to_xy_minmaxs(contours)

		write_xml(msk_path, boxes, xml_dir)

	print('Successfully saved xmls at ' + args.xml_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir')
    parser.add_argument('--msk_dir')
    parser.add_argument('--xml_dir')
    args = parser.parse_args()

    if not os.path.isdir(args.xml_dir):
    	os.mkdir(args.xml_dir)

    main(args.img_dir, args.msk_dir, args.xml_dir)


