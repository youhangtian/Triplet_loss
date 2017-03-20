import os
import numpy as np 

import caffe
import cv2


import lmdb

class sampledata_lmdb():
	def __init__(self, lmdb_path):
		self._sample_table = {}

		self._cursor = lmdb.open(lmdb_path, readonly=True).begin().cursor()
		self._datum = caffe.proto.caffe_pb2.Datum()

		count = 0
		for (index, (key, value)) in enumerate(self._cursor):
			self._datum.ParseFromString(value)
			label = self._datum.label

			if label not in self._sample_table.keys():
				self._sample_table[label] = []

			self._sample_table[label].append([key, 0])
			self._sample_table[label].append([key, 1])

			count += 1
			if (count % 10000 == 0):
				print 'sample count:', count

		print 'Number of classes:', len(self._sample_table)
		print 'Number of training images:', count

	def get_image(self, key):
		value = self._cursor.get(key)
		self._datum.ParseFromString(value)
		label = self._datum.label

		img = caffe.io.datum_to_array(self._datum)
		
		channel_swap = (1, 2, 0)
		img = img.transpose(channel_swap)

		return img, label


class sampledata_disk():
	def __init__(self, folders_path):
		self._sample_table = {}
		self._sample_label = {}

		count = 0

		with open(folders_path) as f:
			for line in f.readlines():
				line = line.strip()
				arr = line.split()

				p = arr[0]
				label = int(arr[1])

				if not self._sample_table.has_key(label):
					self._sample_table[label] = []

				self._sample_table[label].append([p, 0])
				self._sample_table[label].append([p, 1])

				self._sample_label[p] = label

				count += 1
				if (count % 100000 == 0):
					print 'sample count: ', count

		print 'Number of classes:', len(self._sample_table)
		print 'Number of training images:', count, len(self._sample_label)

	def get_image(self, path):
		img = cv2.imread(path, 1)
		label = self._sample_label[path]

		return img, label
