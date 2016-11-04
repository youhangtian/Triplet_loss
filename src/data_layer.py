import caffe
import numpy as np 
import cv2
import copy
import config
import img_process

class DataLayer(caffe.Layer):
	def _shuffle_data(self):
		sample = []
		sample_table = copy.deepcopy(self._sampledata._sample_table)

		for key in sample_table.keys():
			np.random.shuffle(sample_table[key])

		while len(sample_table) > 0:
			key = np.random.choice(sample_table.keys())
			if (len(sample_table[key]) == 0):
				sample_table.pop(key)
				continue

			while (len(sample_table[key]) < self._cut_size):
				randint = np.random.randint(0, len(self._sampledata._sample_table[key]))
				sample_table[key].append(self._sampledata._sample_table[key][randint])

			for i in xrange(self._cut_size):
				sample.append(sample_table[key].pop())

		self._sample = sample

	def _get_next_batch(self):
		if (self._index + self._batch_size > len(self._sample)):
			self._shuffle_data()
			self._index = 0

			self._epoch += 1
			print 'Epoch:', self._epoch

		sample = self._sample[self._index:self._index + self._batch_size]
		self._index += self._batch_size

		imgs_blob = []
		labels_blob = []

		for i in xrange(self._batch_size):
			img, label = self._sampledata.get_image(sample[i][0])
			if (sample[i][1] == 1):
				img = img[:, ::-1, :]

			labels_blob.append(label)
			imgs_blob.append(img_process.img_to_blob(img))

		return np.array(imgs_blob), np.array(labels_blob)

	def set_data(self, sampledata):
		self._sampledata = sampledata

		print 'Epoch:', self._epoch
		self._shuffle_data()

	def setup(self, bottom, top):
		self._index = 0
		self._epoch = 1

		self._batch_size = config.BATCH_SIZE 
		self._cut_size = config.CUT_SIZE 

		top[0].reshape(self._batch_size, 3, 224, 224)
		top[1].reshape(self._batch_size)

	def forward(self, bottom, top):
		imgs_blob, labels_blob = self._get_next_batch()

		top[0].data[...] = imgs_blob
		top[1].data[...] = labels_blob

	def backward(self, top, propagate_down, bottom):
		pass

	def reshape(self, bottom, top):
		pass
