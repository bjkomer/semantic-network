# Leaving the coarse and fine labels as separate for back propagation

from __future__ import print_function

pretrain = False#True # if the model should load pretrained weights
pretrain_name = 'hierarchy_split_rmsprop_mse_e300_aFalse'#'2output_original_rmsprop_categorical_crossentropy_e7_aFalse'

training = True # if the network should train, or just load the weights from elsewhere
optimizer = 'rmsprop'
model_style = 'split'#'original'#'nodroporiginal'#'original'#'split'#'wider'
nb_epoch = 10#50#500#500
learning_rate = 0.01#0.01
data_augmentation = False#True
objective = 'categorical_crossentropy'#'mse' # objective function to use
model_name = '%s_%s_%s_e%s_a%s' % (model_style, optimizer, objective, nb_epoch, data_augmentation)
if optimizer == 'sgd':
    model_name += '_lr%s' % learning_rate
if pretrain:
    model_name += '_pre'
gpu = 'gpu1'

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=%s,floatX=float32" % gpu
from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Graph
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

batch_size = 32
nb_classes_fine = 100
nb_classes_coarse = 20

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
(_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print('y_train_fine shape:', y_train_fine.shape)
print('y_train_coarse shape:', y_train_coarse.shape)

# convert class vectors to binary class matrices
Y_train_fine = np_utils.to_categorical(y_train_fine, nb_classes_fine)
Y_train_coarse = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
Y_test_fine = np_utils.to_categorical(y_test_fine, nb_classes_fine)
Y_test_coarse = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)
print('Y_train_fine shape:', Y_train_fine.shape)
print('Y_train_coarse shape:', Y_train_coarse.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(model_name)

Y_train = np.concatenate((Y_train_coarse, Y_train_fine), axis=1)
Y_test = np.concatenate((Y_test_coarse, Y_test_fine), axis=1)

model = Graph()

model.add_input(name='input', input_shape=(img_channels, img_rows, img_cols))

if model_style == 'original':

    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu3', input='conv3')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4', input='relu3')
    model.add_node(Activation('relu'),
                   name='relu4', input='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2', input='relu4')
    model.add_node(Dropout(0.25),
                   name='drop2', input='pool2')

    model.add_node(Flatten(),
                   name='flat1', input='drop2')
    model.add_node(Dense(512),
                   name='dense1', input='flat1')
    model.add_node(Activation('relu'),
                   name='relu5', input='dense1')
    model.add_node(Dropout(0.5),
                   name='drop3', input='relu5')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='drop3')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='drop3')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')
if model_style == 'nodroporiginal':

    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3', input='pool1')
    model.add_node(Activation('relu'),
                   name='relu3', input='conv3')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4', input='relu3')
    model.add_node(Activation('relu'),
                   name='relu4', input='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2', input='relu4')

    model.add_node(Flatten(),
                   name='flat1', input='pool2')
    model.add_node(Dense(512),
                   name='dense1', input='flat1')
    model.add_node(Activation('relu'),
                   name='relu5', input='dense1')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='relu5')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='relu5')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')
elif model_style == 'moredense':

    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu3', input='conv3')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4', input='relu3')
    model.add_node(Activation('relu'),
                   name='relu4', input='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2', input='relu4')
    model.add_node(Dropout(0.25),
                   name='drop2', input='pool2')

    model.add_node(Flatten(),
                   name='flat1', input='drop2')
    model.add_node(Dense(512),
                   name='dense1', input='flat1')
    model.add_node(Activation('relu'),
                   name='relu5', input='dense1')
    model.add_node(Dropout(0.25),
                   name='drop3', input='relu5')
    model.add_node(Dense(512),
                   name='dense2', input='drop3')
    model.add_node(Activation('relu'),
                   name='relu6', input='dense2')
    model.add_node(Dropout(0.25),
                   name='drop4', input='relu6')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='drop4')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='drop4')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')
elif model_style == 'wider':

    model.add_node(Convolution2D(48, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(48, 3, 3),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(96, 3, 3, border_mode='same'),
                   name='conv3', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu3', input='conv3')
    model.add_node(Convolution2D(96, 3, 3),
                   name='conv4', input='relu3')
    model.add_node(Activation('relu'),
                   name='relu4', input='conv4')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2', input='relu4')
    model.add_node(Dropout(0.25),
                   name='drop2', input='pool2')

    model.add_node(Flatten(),
                   name='flat1', input='drop2')
    model.add_node(Dense(1024),
                   name='dense1', input='flat1')
    model.add_node(Activation('relu'),
                   name='relu5', input='dense1')
    model.add_node(Dropout(0.5),
                   name='drop3', input='relu5')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='drop3')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='drop3')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')
elif model_style == 'custom1':

    model.add_node(Convolution2D(48, 5, 5, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(48, 5, 5),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')
    model.add_node(Dropout(0.10),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(96, 3, 3, border_mode='same'),
                   name='conv3', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu3', input='conv3')
    model.add_node(Convolution2D(96, 3, 3),
                   name='conv4', input='relu3')
    model.add_node(Activation('relu'),
                   name='relu4', input='conv4')
    model.add_node(MaxPooling2D(pool_size=(3, 3)),
                   name='pool2', input='relu4')
    model.add_node(Dropout(0.10),
                   name='drop2', input='pool2')

    model.add_node(Flatten(),
                   name='flat1', input='drop2')
    model.add_node(Dense(1024),
                   name='dense1', input='flat1')
    model.add_node(Activation('relu'),
                   name='relu5', input='dense1')
    model.add_node(Dropout(0.10),
                   name='drop3', input='relu5')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='drop3')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='drop3')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')
elif model_style == 'split': # have some convolutions for each of coarse and fine only

    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3_c', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu3_c', input='conv3_c')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4_c', input='relu3_c')
    model.add_node(Activation('relu'),
                   name='relu4_c', input='conv4_c')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2_c', input='relu4_c')
    model.add_node(Dropout(0.25),
                   name='drop2_c', input='pool2_c')

    model.add_node(Flatten(),
                   name='flat1_c', input='drop2_c')
    model.add_node(Dense(512),
                   name='dense1_c', input='flat1_c')
    model.add_node(Activation('relu'),
                   name='relu5_c', input='dense1_c')
    model.add_node(Dropout(0.5),
                   name='drop3_c', input='relu5_c')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='drop3_c')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3_f', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu3_f', input='conv3_f')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4_f', input='relu3_f')
    model.add_node(Activation('relu'),
                   name='relu4_f', input='conv4_f')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2_f', input='relu4_f')
    model.add_node(Dropout(0.25),
                   name='drop2_f', input='pool2_f')

    model.add_node(Flatten(),
                   name='flat1_f', input='drop2_f')
    model.add_node(Dense(512),
                   name='dense1_f', input='flat1_f')
    model.add_node(Activation('relu'),
                   name='relu5_f', input='dense1_f')
    model.add_node(Dropout(0.5),
                   name='drop3_f', input='relu5_f')
    
    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='drop3_f')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')
elif model_style == 'nodrop_split': # have some convolutions for each of coarse and fine only

    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv2', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu2')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3_c', input='pool1')
    model.add_node(Activation('relu'),
                   name='relu3_c', input='conv3_c')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4_c', input='relu3_c')
    model.add_node(Activation('relu'),
                   name='relu4_c', input='conv4_c')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2_c', input='relu4_c')

    model.add_node(Flatten(),
                   name='flat1_c', input='pool2_c')
    model.add_node(Dense(512),
                   name='dense1_c', input='flat1_c')
    model.add_node(Activation('relu'),
                   name='relu5_c', input='dense1_c')

    #model.add_node(Dense(nb_classes_coarse + nb_classes_fine),
    #               name='dense2', input='drop3')
    model.add_node(Dense(nb_classes_coarse),
                   name='dense_c', input='relu5_c')
    model.add_node(Activation('softmax'),
                   name='soft_c', input='dense_c')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3_f', input='pool1')
    model.add_node(Activation('relu'),
                   name='relu3_f', input='conv3_f')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv4_f', input='relu3_f')
    model.add_node(Activation('relu'),
                   name='relu4_f', input='conv4_f')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2_f', input='relu4_f')

    model.add_node(Flatten(),
                   name='flat1_f', input='pool2_f')
    model.add_node(Dense(512),
                   name='dense1_f', input='flat1_f')
    model.add_node(Activation('relu'),
                   name='relu5_f', input='dense1_f')
    
    model.add_node(Dense(nb_classes_fine),
                   name='dense_f', input='relu5_f')
    model.add_node(Activation('softmax'),
                   name='soft_f', input='dense_f')

model.add_output(name='output_fine', input='soft_f')
model.add_output(name='output_coarse', input='soft_c')

if optimizer == 'sgd':
    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss={'output_fine':objective,'output_coarse':objective}, optimizer=sgd)
else:
    model.compile(loss={'output_fine':objective,'output_coarse':objective}, optimizer=optimizer)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if pretrain:
    model.load_weights('net_output/%s_weights.h5' % pretrain_name)

if training:
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit({'input':X_train, 'output_fine':Y_train_fine,'output_coarse':Y_train_coarse}, batch_size=batch_size,
                  nb_epoch=nb_epoch, #show_accuracy=True,
                  validation_data={'input':X_test, 'output_fine':Y_test_fine,'output_coarse':Y_test_coarse}, shuffle=True)
    else:
        print('Using real-time data augmentation.')

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

    model.save_weights('net_output/2output_%s_weights.h5' % model_name)
    json_string = model.to_json()
    open('net_output/2output_%s_architecture.json' % model_name, 'w').write(json_string)
    if pretrain:
        history.history['pretrain_name'] = pretrain_name
    pickle.dump(history.history, open('net_output/2output_%s_history.p' % model_name,'w'))
    print("saving to: 2output_%s" % model_name)
else:
    #model.load_weights('net_output/keras_cifar100_%s_weights.h5' % model_name)
    model.load_weights('net_output/2output_%s_weights.h5' % model_name)
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
