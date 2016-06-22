# Show the accuracy of a pretrained network

from __future__ import print_function

gpu = 'gpu3'
nb_dim=200

import sys
if len(sys.argv) == 3:
    model_name = sys.argv[1]
    gpu = sys.argv[2]
elif len(sys.argv) == 2:
    model_name = sys.argv[1]
else:
    model_name = 'hierarchy_wider_rmsprop_e50_aFalse'

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
from network_utils import accuracy, accuracy_hierarchy, clean_hierarchy_vec, clean_vec, get_w2v_labels, accuracy_w2v, accuracy_w2v_coarse, classes

# Open an IPython session if an exception is found
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

fname_json = 'net_output/' + model_name + '_architecture.json'
fname_weights = 'net_output/' + model_name + '_weights.h5'


batch_size = 32
nb_classes_fine = 100
nb_classes_coarse = 20

model = model_from_json(open(fname_json).read())
model.load_weights(fname_weights)

def report_two_output_accuracy(X, y_fine, y_coarse, prefix_string='2output hierarchy'):
    Y_fine = np_utils.to_categorical(y_fine, nb_classes_fine)
    Y_coarse = np_utils.to_categorical(y_coarse, nb_classes_coarse)
    
    Y = np.concatenate((Y_coarse, Y_fine), axis=1)

    # Test the model
    Y_predict_dict = model.predict({'input':X}, batch_size=batch_size, verbose=1)

    Y_predict_fine = Y_predict_dict['output_fine']
    Y_predict_coarse = Y_predict_dict['output_coarse']

    Y_predict = np.concatenate((Y_predict_coarse, Y_predict_fine), axis=1)

    # Convert floating point vector to a clean binary vector with only two 1's
    Y_predict_clean = clean_hierarchy_vec(Y_predict)
    
    accuracy, acc_coarse, acc_fine = accuracy_hierarchy(Y_predict_clean, Y)
    print("%s accuracy: %f" % (prefix_string, accuracy))
    print("%s coarse accuracy: %f" % (prefix_string, acc_coarse))
    print("%s fine accuracy: %f" % (prefix_string, acc_fine))

def report_w2v_accuracy(X, y, nb_dim, prefix_string='w2v'):
    Y = get_w2v_labels(y, dim=nb_dim)

    Y_predict = model.predict(X, batch_size=batch_size, verbose=1)

    accuracy, class_, accuracy_c, class_c, accuracy_c2, class_c2 = accuracy_w2v_coarse(Y_predict, Y, dim=nb_dim)

    """
    print(np.sum(class_,axis=0))
    print(np.argmax(np.sum(class_,axis=0)))
    print(classes[np.argmax(np.sum(class_,axis=0))])
    """

    print("%s accuracy: %f" % (prefix_string, accuracy))
    print("%s coarse accuracy: %f" % (prefix_string, accuracy_c))
    print("%s coarse v2 accuracy: %f" % (prefix_string, accuracy_c2))

if 'hierarchy' in model_name:
    # Load and format data
    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
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
elif '2output' in model_name or 'keras_cifar100' in model_name:
    # Load and format data
    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    #TODO: replace the following things with the convenience function

    Y_train_fine = np_utils.to_categorical(y_train_fine, nb_classes_fine)
    Y_train_coarse = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
    Y_test_fine = np_utils.to_categorical(y_test_fine, nb_classes_fine)
    Y_test_coarse = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)
    
    Y_train = np.concatenate((Y_train_coarse, Y_train_fine), axis=1)
    Y_test = np.concatenate((Y_test_coarse, Y_test_fine), axis=1)

    # Test the model
    Y_predict_test_dict = model.predict({'input':X_test}, batch_size=batch_size, verbose=1)
    Y_predict_train_dict = model.predict({'input':X_train}, batch_size=batch_size, verbose=1)

    Y_predict_test_fine = Y_predict_test_dict['output_fine']
    Y_predict_test_coarse = Y_predict_test_dict['output_coarse']

    Y_predict_train_fine = Y_predict_train_dict['output_fine']
    Y_predict_train_coarse = Y_predict_train_dict['output_coarse']

    Y_predict_test = np.concatenate((Y_predict_test_coarse, Y_predict_test_fine), axis=1)
    Y_predict_train = np.concatenate((Y_predict_train_coarse, Y_predict_train_fine), axis=1)

    # Convert floating point vector to a clean binary vector with only two 1's
    Y_predict_test_clean = clean_hierarchy_vec(Y_predict_test)
    Y_predict_train_clean = clean_hierarchy_vec(Y_predict_train)
    
    test_accuracy, test_acc_coarse, test_acc_fine = accuracy_hierarchy(Y_predict_test_clean, Y_test)
    print("2output hierarchy test accuracy: %f" % test_accuracy)
    print("2output hierarchy test coarse accuracy: %f" % test_acc_coarse)
    print("2output hierarchy test fine accuracy: %f" % test_acc_fine)
    
    train_accuracy, train_acc_coarse, train_acc_fine = accuracy_hierarchy(Y_predict_train_clean, Y_train)
    print("2output hierarchy train accuracy: %f" % train_accuracy)
    print("2output hierarchy train coarse accuracy: %f" % train_acc_coarse)
    print("2output hierarchy train fine accuracy: %f" % train_acc_fine)

    # For generalization test, show the accuracy on the things it was not trained on
    if '_gen' in model_name:
        # Indices of the things it was trained on
        indices_base = y_train_fine[y_train_fine % 5 != 0]
        y_train_fine_base = y_train_fine[indices_base]
        y_train_coarse_base = y_train_coarse[indices_base]
        X_train_base = X_train[indices_base]
        
        # Indices of the things it was not trained on
        indices_gen = y_train_fine[y_train_fine % 5 == 0]
        y_train_fine_gen = y_train_fine[indices_gen]
        y_train_coarse_gen = y_train_coarse[indices_gen]
        X_train_gen = X_train[indices_gen]
        
        indices_base_test = y_test_fine[y_test_fine % 5 != 0]
        y_test_fine_base = y_test_fine[indices_base_test]
        y_test_coarse_base = y_test_coarse[indices_base_test]
        X_test_base = X_test[indices_base_test]
        
        indices_gen_test = y_test_fine[y_test_fine % 5 == 0]
        y_test_fine_gen = y_test_fine[indices_gen_test]
        y_test_coarse_gen = y_test_coarse[indices_gen_test]
        X_test_gen = X_test[indices_gen_test]

        report_two_output_accuracy(X_train_gen, y_train_fine_gen, y_train_coarse_gen, "2output hierarchy train gen")
        report_two_output_accuracy(X_test_gen, y_test_fine_gen, y_test_coarse_gen, "2output hierarchy test gen")
        
        report_two_output_accuracy(X_train_base, y_train_fine_base, y_train_coarse_base, "2output hierarchy train base")
        report_two_output_accuracy(X_test_base, y_test_fine_base, y_test_coarse_base, "2output hierarchy test base")

elif 'w2v' in model_name:
    # Load and format the data
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    # Figure out how many dimensions were used
    for d in [200,100,50,25,10]:
	if 'd%s'%d in model_name:
	    nb_dim = d
	    break
    Y_train = get_w2v_labels(y_train, dim=nb_dim)
    Y_test = get_w2v_labels(y_test, dim=nb_dim)

    #NOTE: due to changes in keras, model has to be compiled before being used here
    #TODO: do the parameters that it is compiled with matter? might need to get them exactly
    optimizers = ['sgd','rmsprop','adam']
    losses = ['mse','msle']
    model.compile(optimizer=optimizers[1],loss=losses[0])

    Y_predict_test = model.predict(X_test, batch_size=batch_size, verbose=1)
    Y_predict_train = model.predict(X_train, batch_size=batch_size, verbose=1)

    #test_accuracy, test_class = accuracy_w2v(Y_predict_test, Y_test, dim=nb_dim)
    test_accuracy, test_class, test_accuracy_c, test_class_c, test_accuracy_c2, test_class_c2 = accuracy_w2v_coarse(Y_predict_test, Y_test, dim=nb_dim)

    #train_accuracy, train_class = accuracy_w2v(Y_predict_train, Y_train, dim=nb_dim)
    train_accuracy, train_class, train_accuracy_c, train_class_c, train_accuracy_c2, train_class_c2 = accuracy_w2v_coarse(Y_predict_train, Y_train, dim=nb_dim)

    #sanity_accuracy, sanity_class = accuracy_w2v(Y_test, Y_test)

    print(np.sum(test_class,axis=0))
    print(np.argmax(np.sum(test_class,axis=0)))
    print(classes[np.argmax(np.sum(test_class,axis=0))])
    print(np.sum(train_class,axis=0))
    print(np.argmax(np.sum(train_class,axis=0)))
    print(classes[np.argmax(np.sum(train_class,axis=0))])
    #print(np.sum(sanity_class,axis=0))

    print("w2v test accuracy: %f" % test_accuracy)
    print("w2v train accuracy: %f" % train_accuracy)
    print("w2v test coarse accuracy: %f" % test_accuracy_c)
    print("w2v train coarse accuracy: %f" % train_accuracy_c)
    print("w2v test coarse v2 accuracy: %f" % test_accuracy_c2)
    print("w2v train coarse v2 accuracy: %f" % train_accuracy_c2)
     
    # Sanity checks
    report_w2v_accuracy(X_train, y_train, nb_dim, "w2v train sanity")
    report_w2v_accuracy(X_test, y_test, nb_dim, "w2v test sanity")
    
    # For generalization test, show the accuracy on the things it was not trained on
    if '_gen' in model_name:
        # Indices of the things it was trained on
        indices_base = y_train[y_train % 5 != 0]
        y_train_base = y_train[indices_base]
        X_train_base = X_train[indices_base]
        
        # Indices of the things it was not trained on
        indices_gen = y_train[y_train % 5 == 0]
        y_train_gen = y_train[indices_gen]
        X_train_gen = X_train[indices_gen]
        
        indices_base_test = y_test[y_test % 5 != 0]
        y_test_base = y_test[indices_base_test]
        X_test_base = X_test[indices_base_test]
        
        indices_gen_test = y_test[y_test % 5 == 0]
        y_test_gen = y_test[indices_gen_test]
        X_test_gen = X_test[indices_gen_test]

        report_w2v_accuracy(X_train_gen, y_train_gen, nb_dim, "w2v train gen")
        report_w2v_accuracy(X_test_gen, y_test_gen, nb_dim, "w2v test gen")
        
        report_w2v_accuracy(X_train_base, y_train_base, nb_dim, "w2v train base")
        report_w2v_accuracy(X_test_base, y_test_base, nb_dim, "w2v test base")
        

elif 'coarse' in model_name or 'fine' in model_name:
    # Load and format data
    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    Y_train_fine = np_utils.to_categorical(y_train_fine, nb_classes_fine)
    Y_train_coarse = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
    Y_test_fine = np_utils.to_categorical(y_test_fine, nb_classes_fine)
    Y_test_coarse = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)
 
    if 'coarse' in model_name:
        Y_train = Y_train_coarse
        Y_test = Y_test_coarse
    elif 'fine' in model_name:
        Y_train = Y_train_fine
        Y_test = Y_test_fine
    print(X_test)
    # Test the model
    Y_predict_test = model.predict(X_test, batch_size=batch_size, verbose=1)
    Y_predict_train = model.predict(X_train, batch_size=batch_size, verbose=1)
    #print(Y_predict_train)
    # Convert floating point vector to a clean binary vector with only two 1's
    Y_predict_test_clean = clean_vec(Y_predict_test)
    Y_predict_train_clean = clean_vec(Y_predict_train)
    
    if 'coarse' in model_name:
        label_type = 'coarse'
    elif 'fine' in model_name:
        label_type = 'fine'

    test_accuracy = accuracy(Y_predict_test_clean, Y_test)
    print(label_type + " test accuracy: %f" % test_accuracy)
    
    train_accuracy = accuracy(Y_predict_train_clean, Y_train)
    print(label_type + " train accuracy: %f" % train_accuracy)
else:
    print("Unsure which accuracy measure to use based on file name")
