from __future__ import print_function
gpu = 'gpu1'
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=%s,floatX=float32" % gpu
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, quniform
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import cPickle as pickle

# Open an IPython session if an exception is found
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

nb_epoch = 10 #NOTE: need to modify this elsewhere as well
nb_evals = 50

def data():

    nb_classes_fine = 100
    nb_classes_coarse = 20

    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
    
    Y_train = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
    Y_test = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return X_train, Y_train, X_test, Y_test
    """   
    Y_train_fine = np_utils.to_categorical(y_train_fine, nb_classes_fine)
    Y_train_coarse = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
    Y_test_fine = np_utils.to_categorical(y_test_fine, nb_classes_fine)
    Y_test_coarse = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)

    return X_train, Y_train_fine, X_test, Y_test_fine
    """
def model(X_train, Y_train, X_test, Y_test):

    nb_dim = 20
    img_rows, img_cols = 32, 32
    img_channels = 3

    dense_layer_size = {{choice([256, 512, 1024])}}
    optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
    batch_size = {{choice([32, 64, 128])}}
    num_conv1 = int({{quniform(24, 64, 1)}})
    num_conv2 = int({{quniform(32, 96, 1)}})
    params = {'dense_layer_size':dense_layer_size,
              'optimizer':optimizer,
              'batch_size':batch_size,
              'num_conv1':num_conv1,
              'num_conv2':num_conv2,
             }


    model = Sequential()

    model.add(Convolution2D(num_conv1, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_conv1, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(num_conv2, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_conv2, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(dense_layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_dim))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    model.fit(X_train, Y_train,
             batch_size=batch_size,
             nb_epoch=10,
             show_accuracy=True,
             verbose=2,
             validation_data=(X_test, Y_test))

    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    #return {'loss': -acc, 'status': STATUS_OK, 'model':model}
    return {'loss': -acc, 'status': STATUS_OK, 'params':params}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=nb_evals,
                                          trials=trials)

    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    
    pickle.dump(trials, open('net_output/trials_coarse_epoch%s_evals%s.p'%(nb_epoch, nb_evals)))
