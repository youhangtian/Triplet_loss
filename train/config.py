#training config
CAFFEPATH = '/home/xiaotian/caffe'

SOLVER = '../models/inception21k_conv/inception21k_solver.prototxt'
TRAINED_FILE = 'Inception21k.caffemodel'
USE_GPU = 1
DEVICE_ID = 1

USE_LMDB = 0
LMDB_PATH = 'path/to/train_lmdb'
DATA_FILE = 'filename'

#triplet config
BATCH_SIZE = 50
CUT_SIZE = 5

IMG_WIDTH = 224
IMG_HEIGHT = 224

MARGIN = 0.2
