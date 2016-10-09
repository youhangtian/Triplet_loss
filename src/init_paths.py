import os.path as osp
import sys
import config

def add_path(path):
	if path not in sys.path:
		sys.path.insert(0, path)

caffe_path = config.CAFFEPATH + '/python'
add_path(caffe_path)

lib_path = '../'
add_path(lib_path)

triplet_path = '../triplet_loss'
add_path(triplet_path)