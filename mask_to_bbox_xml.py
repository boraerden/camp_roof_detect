from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import time
from pprint import pprint
from pdb import set_trace as t
import xml.etree.cElementTree as ET
from tqdm import tqdm

IMAGE_DIR_PATH = '../new_imgs_mks/newimages/'
MASK_DIR_PATH = '../new_imgs_mks/newmasks/'
XML_FOLDER = '../retinanet-data-1.5/xmls/'

def write_xml(msk_path, boxes):
	write_path = XML_FOLDER + msk_path.split('/')[-1][:-4] + '.xml'


	annotation = ET.Element("annotation")
	folder = ET.SubElement(annotation, "folder").text = IMAGE_DIR_PATH
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



# Create list of paths for images and masks
db_imgs = glob(IMAGE_DIR_PATH + '*.png')
db_msks = glob(MASK_DIR_PATH + '*.png')

# Prune pre-augmented images and masks
image_paths = [img for img in db_imgs if 'm90' not in img and 'p90' not in img]
mask_paths = [msk for msk in db_msks if 'm90' not in msk and 'p90' not in msk]

	

for img_path, msk_path in tqdm(zip(image_paths, mask_paths)):
	msk = np.asarray(Image.open(msk_path))

	width, height = msk.shape

	boxes = []
	in_box = []
	for y in range(height):
		for x in range(width):


			# if start detecting segmentation label
			if msk[y,x] != 0 and (x,y) not in in_box:

				# go right until label is done
				i = 0
				while x+i < width and msk[y,x+i] != 0:
					i = i+1

				# go down until label is done
				j=0
				while y+j<height and msk[y+j,x] != 0:
					j = j+1

				# deduce bouding box
				xmin = x
				xmax = x + i - 1
				ymin = y
				ymax = y + j - 1

				# mark pixels as done to skip in double for loop
				for a in range(xmin, xmax+1):
					for b in range(ymin, ymax+1):
						in_box.append((a,b))

				# add box to list 
				# making it 5 pixels larger on each side
				# while remaining in image
				boxes.append((max(0,xmin-5),
								max(0, ymin-5),
								min(width, xmax+5),
								min(height, ymax+5)))



	write_xml(msk_path, boxes)
	# img = np.asarray(Image.open(img_path))
	# plt.imshow(img)
	# plt.imshow(msk, alpha=.4)
	# plt.show()
	# plt.close()
	# time.sleep(1.5)
