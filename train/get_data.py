import os
import sys
import numpy as numpy

if (len(sys.argv) != 3):
	print 'Usage:', str(sys.argv[0]), 'folders_path', 'out_file'
	exit()

folders_path = str(sys.argv[1])
outfile = str(sys.argv[2])

res = []
table = {}

for num, folder_name in enumerate(sorted(os.listdir(folders_path))):
	img_names = os.listdir(folders_path + '/' + folder_name)

	label = num
	
	for i in xrange(len(img_names)):
		path = folders_path + '/' + folder_name + '/' + img_names[i]

		res.append('{} {}'.format(path, label))

with open(outfile, 'w') as f:
	for i in xrange(len(res)):
		f.write('{}\n'.format(res[i]))

print 'Down!'