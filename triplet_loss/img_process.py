import numpy as np 
import cv2

def img_to_blob(img):
	target_size = 224
	means = np.array([[[128, 128, 128]]])

	img = img.astype(np.float32, copy=False)
	img -= means

	img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

	blob = np.copy(img)

	channel_swap = (2, 0, 1)
	blob = blob.transpose(channel_swap)
	return blob
	