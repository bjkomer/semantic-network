from __future__ import print_function
gpu = 'gpu0'
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=%s,floatX=float32" % gpu
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, quniform
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from network_utils import get_w2v_labels, accuracy_w2v
import numpy as np

import cPickle as pickle

# Open an IPython session if an exception is found
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

nb_epoch = 15#35#50#10 #NOTE: need to modify this elsewhere as well
nb_evals = 120#150#50
nb_dim = 50 #NOTE: this needs to be modified in two other places as well

def data():

    nb_dim=50
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

    # convert class vectors to binary class matrices
    Y_train = get_w2v_labels(y_train, dim=nb_dim)
    Y_test = get_w2v_labels(y_test, dim=nb_dim)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    img_rows, img_cols = 32, 32
    img_channels = 3
    nb_dim = 50
    nb_epoch=15#35#30

    dense_layer_size = {{choice([256, 512, 1024])}}
    objective = 'mse'
    #optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
    optimizer = {{choice(['rmsprop', 'sgd'])}}
    batch_size = {{choice([32, 64, 128])}}
    num_conv1 = int({{quniform(24, 64, 1)}})
    num_conv2 = int({{quniform(32, 96, 1)}})
    size_conv1 = int({{quniform(2, 5, 1)}})
    size_conv2 = int({{quniform(2, 5, 1)}})
    early_dropout = {{uniform(0,.75)}}
    late_dropout = {{uniform(0,.75)}}
    data_augmentation = {{choice(['True','False'])}}
    final_activation = {{choice(['none','linear'])}}
    params = {'dense_layer_size':dense_layer_size,
              'optimizer':optimizer,
              'batch_size':batch_size,
              'num_conv1':num_conv1,
              'num_conv2':num_conv2,
              'size_conv1':size_conv1,
              'size_conv2':size_conv2,
              'final_activation':final_activation,
              'early_dropout':early_dropout,
              'late_dropout':late_dropout
             }
    if optimizer == 'sgd':
        learning_rate = {{loguniform(np.log(0.001),np.log(0.999))}}
        params['learning_rate'] = learning_rate

    if data_augmentation:
        more_augmentation = {{choice(['True','False'])}}
        params['more_augmentation'] = more_augmentation

    model = Sequential()


    model.add(Convolution2D(num_conv1, size_conv1, size_conv1, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_conv1, size_conv1, size_conv1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(early_dropout))

    model.add(Convolution2D(num_conv2, size_conv2, size_conv2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_conv2, size_conv2, size_conv2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(early_dropout))

    model.add(Flatten())
    model.add(Dense(dense_layer_size))
    model.add(Activation('relu'))
    model.add(Dropout(late_dropout))
    model.add(Dense(nb_dim))

    if final_activation != 'none':
        model.add(Activation(final_activation))

    if optimizer == 'sgd':
        # let's train the model using SGD + momentum (how original).
        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=objective, optimizer=sgd)
    elif optimizer == 'rmsprop':
        model.compile(loss=objective, optimizer='rmsprop')
    else:
        model.compile(loss=objective, optimizer=optimizer)

    print(params)

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(X_train, Y_train, batch_size=batch_size,
                  nb_epoch=nb_epoch, show_accuracy=True,
                  validation_data=(X_test, Y_test), shuffle=True)
    else:
        print('Using real-time data augmentation.')
        if more_augmentation:
            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=True,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=True,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
        else:
            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch, show_accuracy=True,
                            validation_data=(X_test, Y_test),
                            nb_worker=1)

    #score, acc = model.evaluate(X_test, Y_test, verbose=0)
    loss = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)

    return {'loss': loss, 'status': STATUS_OK, 'params':params}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=nb_evals,
                                          trials=trials)

    X_train, Y_train, X_test, Y_test = data()
    #print("Evaluation of best performing model:")
    #print(best_model.evaluate(X_test, Y_test))

    pickle.dump(trials, open('net_output/trials_w2v_dim%s_epoch%s_evals%s.p'%(nb_dim, nb_epoch, nb_evals),'w'))
