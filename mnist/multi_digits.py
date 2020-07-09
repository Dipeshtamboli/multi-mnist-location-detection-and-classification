import pdb
from tqdm import tqdm
import pickle, gzip
import numpy as np
import pdb
import cv2
import os
import imutils
from random import sample,randint
f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

txt_f = open("val_data.csv", "w")
i_iter = 0
img_count = 1

while (i_iter<len(validation_data[1])):
	print(i_iter)
	num = randint(3,6)
	# 9 placeholders in a 224x224 image
	sequence = [i for i in range(9)]
	subset = sample(sequence, num)
	# print(subset)

	imgs = []
	labs = []
	coordinates = []
	s_labs, s_coordi, s_size, s_rotn = "","","", ""
	frame = np.zeros((224,224))
	for i in ((subset)):
		img = validation_data[0][i_iter]
		img = np.reshape(img,(28,28,1))
		angle = randint(-15,15)
		img = imutils.rotate(img, angle)

		jitter_x = 37 + randint(-25,25)
		jitter_y = 37 + randint(-25,25)
		x = i%3
		y = i//3
		x_coordinate = 74*x+jitter_x
		y_coordinate = 74*y+jitter_y
		coordinates.append((x_coordinate, y_coordinate))

		largest = min(56,2*(224- y_coordinate),2*(224- x_coordinate))
		if largest<28:
			pdb.set_trace()
		size = randint(28,largest)
		img = cv2.resize(img, (size,size))
		lab = validation_data[1][i_iter]
		x_init = x_coordinate-size//2
		if x_init <0: x_init=0
		x_end = x_init + size
		y_init = y_coordinate-size//2
		if y_init <0: y_init=0
		y_end = y_init + size

		frame[x_init:x_end,y_init:y_end] += img
		imgs.append(img)
		labs.append(lab)
		i_iter +=1
		s_size += str(size) + ","
		s_coordi += str(x_coordinate) +"-"+ str(y_coordinate) + ","
		s_labs += str(lab) + ","
		s_rotn += str(angle) + ","

	filename =f"{img_count}_{num}_{labs}.jpg" 
	filename = filename.replace(', ','-')
	writeline =filename + ","+str(len(imgs)) +","+ s_labs + s_coordi + s_size + s_rotn
	txt_f.write(writeline +"\n")
	# cv2.imshow("frame", frame*255)
	# cv2.waitKey(0)
	cv2.imwrite(f"val_multi_imgs/"+filename,frame*255)
	img_count += 1

	# print(i_iter)
	# print(labs)
	# pdb.set_trace()
	# exit()
txt_f.close()