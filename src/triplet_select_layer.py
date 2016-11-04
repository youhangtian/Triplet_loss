import caffe
import numpy as np 
import config

class TripletSelectLayer(caffe.Layer):
	def setup(self, bottom, top):
		self.margin = config.MARGIN 

	def forward(self, bottom, top):
		bottom_data = np.array(bottom[0].data)
		bottom_label = np.array(bottom[1].data)
		self.index_map = []

		anchor_arr = []
		positive_arr = []
		negative_arr = []

		data_table = {}
		for i in xrange(bottom[0].num):
			label = bottom_label[i]
			if (label not in data_table.keys()):
				data_table[label] = []

			data_table[label].append(i)

		for i in xrange(bottom[0].num):
			anchor_label = bottom_label[i]

			if (len(data_table[anchor_label]) < 2):
				continue;

			for j in xrange(len(data_table[anchor_label])):
				positive_index = data_table[anchor_label][j]
				if (positive_index == i):
					continue

				positive_label = bottom_label[positive_index]
				
				negative_index = positive_index
				negative_label = positive_label

				dist = -1.0

				max_iter = bottom[0].num * 2
				while ((negative_label == positive_label) or (dist < 0)):
					if (max_iter <= 0):
						break;

					negative_label = np.random.choice(data_table.keys())
					negative_index = np.random.choice(data_table[negative_label])
					
					anchor = bottom_data[i]
					positive = bottom_data[positive_index]
					negative = bottom_data[negative_index]
					ap = np.sum((anchor - positive) ** 2)
					an = np.sum((anchor - negative) ** 2)

					dist = ap + self.margin - an

					max_iter -= 1
					

				if (positive_label != negative_label):
					anchor_arr.append(bottom_data[i])
					positive_arr.append(bottom_data[positive_index])
					negative_arr.append(bottom_data[negative_index])

					self.index_map.append([i, positive_index, negative_index])


		top[0].reshape(*np.array(anchor_arr).shape)
		top[1].reshape(*np.array(anchor_arr).shape)
		top[2].reshape(*np.array(anchor_arr).shape)

		top[0].data[...] = np.array(anchor_arr)
		top[1].data[...] = np.array(positive_arr)
		top[2].data[...] = np.array(negative_arr)

	def backward(self, top, propagate_down, bottom):
		if propagate_down[0]:
			bottom_diff = np.zeros(bottom[0].data.shape)

			for i in xrange(top[0].num):
				bottom_diff[self.index_map[i][0]] += top[0].diff[i]
				bottom_diff[self.index_map[i][1]] += top[1].diff[i]
				bottom_diff[self.index_map[i][2]] += top[2].diff[i]

			bottom[0].diff[...] = bottom_diff

	def reshape(self, bottom, top):
		pass