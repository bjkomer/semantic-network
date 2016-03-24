# Show the accuracy of a pretrained network

from __future__ import print_function

gpu = 'gpu0'

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=%s,floatX=float32" % gpu
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Graph, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import cPickle as pickle
import numpy as np
from network_utils import accuracy, accuracy_hierarchy, clean_hierarchy_vec

# Open an IPython session if an exception is found
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = 'hierarchy_wider_rmsprop_e50_aFalse'
fname_json = 'net_output/' + model_name + '_architecture.json'
fname_weights = 'net_output/' + model_name + '_weights.h5'


batch_size = 32
nb_classes_fine = 100
nb_classes_coarse = 20

model = model_from_json(open(fname_json).read())
model.load_weights(fname_weights)

if 'hierarchy' in model_name:
    # Load and format data
    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
    
    Y_train_fine = np_utils.to_categorical(y_train_fine, nb_classes_fine)
    Y_train_coarse = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
    Y_test_fine = np_utils.to_categorical(y_test_fine, nb_classes_fine)
    Y_test_coarse = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)
    
    Y_train = np.concatenate((Y_train_coarse, Y_train_fine), axis=1)
    Y_test = np.concatenate((Y_test_coarse, Y_test_fine), axis=1)

    # Test the model
    Y_predict_test = model.predict({'input':X_test}, batch_size=batch_size, verbose=1)['output']
    Y_predict_train = model.predict({'input':X_train}, batch_size=batch_size, verbose=1)['output']

    # Convert floating point vector to a clean binary vector with only two 1's
    Y_predict_test_clean = clean_hierarchy_vec(Y_predict_test)
    Y_predict_train_clean = clean_hierarchy_vec(Y_predict_train)
    
    test_accuracy, test_acc_coarse, test_acc_fine = accuracy_hierarchy(Y_predict_test_clean, Y_test)
    print("hierarchy test accuracy: %f" % test_accuracy)
    print("hierarchy test coarse accuracy: %f" % test_acc_coarse)
    print("hierarchy test fine accuracy: %f" % test_acc_fine)
    
    train_accuracy, train_acc_coarse, train_acc_fine = accuracy_hierarchy(Y_predict_train_clean, Y_train)
    print("hierarchy train accuracy: %f" % train_accuracy)
    print("hierarchy train coarse accuracy: %f" % train_acc_coarse)
    print("hierarchy train fine accuracy: %f" % train_acc_fine)
