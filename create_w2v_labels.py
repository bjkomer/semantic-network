# View Images in the CIFAR Dataset from the HDF5 format
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import time
import word2vec
import getpass

DIM = 200
NUM_TRAIN = 50000
NUM_TEST = 10000

if len(sys.argv) == 2:
    DIM = int(sys.argv[1])

user = getpass.getuser()
if user == 'ctnuser':
    root = '/home/ctnuser/bjkomer/'
elif user == 'bjkomer':
    root = '/home/bjkomer/'

# TODO: make sure this is the correct order of the labels
classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
           'fish', 'flounder', 'ray', 'shark', 'trout',
           'orchids', 'poppies', 'roses', 'sunflower', 'tulips',
           'bottles', 'bowls', 'cans', 'cups', 'plates',
           'apples', 'mushrooms', 'oranges', 'pears', 'peppers',
           'clock', 'keyboard', 'lamp', 'telephone', 'television',
           'bed', 'chair', 'couch', 'table', 'wardrobe',
           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
           'bear', 'leopard', 'lion', 'tiger', 'wolf',
           'bridge', 'castle', 'house', 'road', 'skyscraper',
           'cloud', 'forest', 'mountain', 'plain', 'sea',
           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
           'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
           'crab', 'lobster', 'snail', 'spider', 'worm',
           'baby', 'boy', 'girl', 'man', 'woman',
           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
           'maple', 'oak', 'palm', 'pine', 'willow',
           'bicycle', 'bus', 'motorcycle', 'truck', 'train',
           'cutter', 'rocket', 'streetcar', 'tank', 'tractor'
          ]

data_prefix = 'data/cifar_100_caffe_hdf5/'
fnametrain = data_prefix + 'train.h5'
fnametest = data_prefix + 'test.h5'
fnametrain_label = data_prefix + 'train_w2v_label.h5'
fnametest_label = data_prefix + 'test_w2v_label.h5'
if DIM == 200:
    model = word2vec.load(root + 'word2vec/vectors.bin')
elif DIM in [100,50,25,10]:
    model = word2vec.load(root + 'semantic-network/data/text8-%s.bin'%DIM)
else:
    raise NotImplemented
"""
# Training
ftrain = h5py.File(fnametrain, 'r')
ftrain_label = h5py.File(fnametrain_label, 'w')
train_label = np.zeros((NUM_TRAIN, DIM))

for i, label in enumerate(ftrain['label_fine']):
    train_label[i,:] = model[classes[label]]

ftrain_label.create_dataset('label_w2v', data=train_label)
ftrain_label.close()

# Testing
ftest = h5py.File(fnametest, 'r')
ftest_label = h5py.File(fnametest_label, 'w')
test_label = np.zeros((NUM_TEST, DIM))

for i, label in enumerate(ftest['label_fine']):
    test_label[i,:] = model[classes[label]]

ftest_label.create_dataset('label_w2v', data=test_label)
ftest_label.close()
"""
# Ordered Label List
fnamelist_label = data_prefix + 'ordered_w2v_labels_%sdim.h5'%DIM
flist_label = h5py.File(fnamelist_label, 'w')
list_label = np.zeros((100, DIM))

for i, c in enumerate(classes):
    list_label[i,:] = model[c]

flist_label.create_dataset('label_w2v', data=list_label)
flist_label.close()
