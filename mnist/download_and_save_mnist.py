from tqdm import tqdm
import pickle, gzip
import numpy as np
import pdb
import cv2
import os
'''
DOWNLOAD LINK
http://deeplearning.net/data/mnist/mnist.pkl.gz
'''
import wget
print('Beginning file download with wget module')
url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
wget.download(url, 'mnist.pkl.gz')

f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
os.makedirs("train")
os.makedirs("test")
os.makedirs("val")
for i in tqdm(range(len(training_data[1]))):
	img = training_data[0][i]
	img = np.reshape(img,(28,28,1))
	lab = training_data[1][i]
	cv2.imwrite(f"train/{lab}_{i}.jpg",img*255)
for i in tqdm(range(len(validation_data[1]))):
	img = validation_data[0][i]
	img = np.reshape(img,(28,28,1))
	lab = validation_data[1][i]
	cv2.imwrite(f"val/{lab}_{i}.jpg",img*255)
for i in tqdm(range(len(test_data[1]))):
	img = test_data[0][i]
	img = np.reshape(img,(28,28,1))
	lab = test_data[1][i]
	cv2.imwrite(f"test/{lab}_{i}.jpg",img*255)
f.close()
