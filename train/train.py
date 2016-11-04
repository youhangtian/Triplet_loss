import init_paths
import config

from src.sampledata import sampledata_lmdb, sampledata_disk

import caffe
import numpy as np 
import os
from caffe.proto import caffe_pb2
import google.protobuf as pb2 

class SolverWrapper(object):
	def __init__(self, solver, trained_file=None, use_gpu=1, device_id=0, sampledata=None):
		if use_gpu:
			caffe.set_mode_gpu()
			caffe.set_device(device_id)
		else:
			caffe.set_mode_cpu()

		self.solver = caffe.SGDSolver(solver)
		if trained_file is not None:
			print 'Loading pretrained model weights from: ', trained_file
			self.solver.net.copy_from(trained_file)

		self.solver_param = caffe_pb2.SolverParameter()
		with open(solver, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)

		self.solver.net.layers[0].set_data(sampledata)

	def train_model(self):
		while(self.solver.iter < self.solver_param.max_iter):
			self.solver.step(1)


if __name__ == '__main__':
	solver = config.SOLVER 
	trained_file = config.TRAINED_FILE 
	use_gpu = config.USE_GPU
	device_id = config.DEVICE_ID

	lmdb_path = config.LMDB_PATH 
	folders_path = config.FOLDERS_PATH 
	use_lmdb = config.USE_LMDB 

	if use_lmdb == 1:
		sampledata = sampledata_lmdb(lmdb_path)
	else:
		sampledata = sampledata_disk(folders_path)

	sw = SolverWrapper(solver, trained_file, use_gpu, device_id, sampledata)

	print 'Solving...'
	sw.train_model()
	print 'Down!'