import argparse
from glob import glob                                                           
import cv2 
import os
from tqdm import tqdm

def png_to_jpg(png_dir, jpg_save_dir):

	pngs = glob(os.path.join(png_dir,'*.png'))

	for png in tqdm(pngs):
	    img = cv2.imread(png)
	    write_path = os.path.join(jpg_save_dir, png.split('/')[-1][:-3] + 'jpg')
	    cv2.imwrite(write_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--png_dir')
    parser.add_argument('--jpg_save_dir')
    args = parser.parse_args()

    png_to_jpg(args.png_dir, args.jpg_save_dir)

    print('Successfully converted pngs to jpgs saved at ' + args.jpg_save_dir)


