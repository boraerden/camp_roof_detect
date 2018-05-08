import numpy as np
from glob import glob
from pdb import set_trace as t
from tqdm import tqdm

IMAGE_DIR_PATH = '../../data_folder/train/img/0/'
write_filename = '../../data_folder_imagesets/imagesets_train.txt'


imgs = glob(IMAGE_DIR_PATH + '*.png')
img_name = [s.split('/')[-1][:-4] for s in imgs]

with open(write_filename, mode="w") as outfile:  # also, tried mode="rb"
    for s in img_name:
        outfile.write(s+'\n')