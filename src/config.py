import os
import numpy as np 

CAFFEPATH = 'path/to/caffe'

SOLVER = 'path/to/vgg_binary_solver.prototxt'
TRAINED_FILE = 'path/to/VGG_FACE.caffemodel'
USE_GPU = 1
DEVICE_ID = 1

LMDB_PATH = 'path/to/train_lmdb'
FOLDERS_PATH = 'path/to/train_folders'
USE_LMDB = 0