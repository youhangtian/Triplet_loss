import caffe
import numpy as np 
import config

class TripletLossLayer(caffe.Layer):
	def setup(self, bottom, top):
		assert bottom[0].num == bottom[1].num, 'bottom[0]:{}, bottom[1]:{}'.format(bottom[0].num, bottom[1].num)
		assert bottom[0].num == bottom[2].num, 'bottom[0]:{}, bottom[1]:{}'.format(bottom[0].num, bottom[2].num)

		self.margin = config.MARGIN 

		top[0].reshape(1)

	def forward(self, bottom, top):
		anchor = np.array(bottom[0].data)
		positive = np.array(bottom[1].data)
		negative = np.array(bottom[2].data)

		if (bottom[0].num <= 0):
			top[0].data[...] = 0.0
		else:
			ap = np.sum((anchor - positive) ** 2, axis=1)
			an = np.sum((anchor - negative) ** 2, axis=1)

			dist = self.margin + ap - an
			dist_hinge = np.maximum(dist, 0.0)

			self.residual_list = []
			for i in xrange(len(dist_hinge)):
				if (dist_hinge[i] > 0.0):
					self.residual_list.append(dist_hinge[i])
				else:
					self.residual_list.append(0.0)
			loss = np.sum(dist_hinge) / bottom[0].num 

			top[0].data[...] = loss

	def backward(self, top, propagate_down, bottom):
		if propagate_down[0]:
			anchor = np.array(bottom[0].data)
			positive = np.array(bottom[1].data)
			negative = np.array(bottom[2].data)

			if (bottom[0].num > 0):
				coeff = 2.0 * top[0].diff / bottom[0].num
				diff_a = coeff * np.dot(np.diag(self.residual_list), (negative - positive))
				diff_p = coeff * np.dot(np.diag(self.residual_list), (positive -  anchor))
				diff_n = coeff * np.dot(np.diag(self.residual_list), (anchor - negative))

				bottom[0].diff[...] = diff_a
				bottom[1].diff[...] = diff_p
				bottom[2].diff[...] = diff_n

	def reshape(self, bottom, top):
		pass