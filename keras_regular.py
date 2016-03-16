# Standard labels for Cifar100 using Keras
from __future__ import print_function
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import cPickle as pickle

batch_size = 32
nb_classes_fine = 100
nb_classes_coarse = 20
nb_epoch = 200
data_augmentation = True

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

######################
# Beginning of Model #
######################
model = Graph()

model.add_input(name='input', input_shape=(img_channels, img_rows, img_cols))

#TODO: make things nice and in loops
model_name = 'conv2'
if model_name = 'conv6':

    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv1-1', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu1-1', input='conv1-1')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu1-1')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv2', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv2-1', input='relu2')
    model.add_node(Activation('relu'),
                   name='relu2-1', input='conv2-1')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2', input='relu2-1')
    model.add_node(Dropout(0.25),
                   name='drop2', input='pool2')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv3', input='drop2')
    model.add_node(Activation('relu'),
                   name='relu3', input='conv3')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv3-1', input='relu3')
    model.add_node(Activation('relu'),
                   name='relu3-1', input='conv3-1')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool3', input='relu3-1')
    model.add_node(Dropout(0.25),
                   name='drop3', input='pool3')
    model.add_node(Flatten(),
                   name='flatten', input='drop3')

elif model_name == 'conv4':
    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv1-1', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu1-1', input='conv1-1')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu1-1')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Convolution2D(64, 3, 3, border_mode='same'),
                   name='conv2', input='drop1')
    model.add_node(Activation('relu'),
                   name='relu2', input='conv2')
    model.add_node(Convolution2D(64, 3, 3),
                   name='conv2-1', input='relu2')
    model.add_node(Activation('relu'),
                   name='relu2-1', input='conv2-1')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool2', input='relu2-1')
    model.add_node(Dropout(0.25),
                   name='drop2', input='pool2')

    model.add_node(Flatten(),
                   name='flatten', input='drop2')

elif model_name == 'conv2':
    model.add_node(Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)),
                   name='conv1', input='input')
    model.add_node(Activation('relu'),
                   name='relu1', input='conv1')
    model.add_node(Convolution2D(32, 3, 3),
                   name='conv1-1', input='relu1')
    model.add_node(Activation('relu'),
                   name='relu1-1', input='conv1-1')
    model.add_node(MaxPooling2D(pool_size=(2, 2)),
                   name='pool1', input='relu1-1')
    model.add_node(Dropout(0.25),
                   name='drop1', input='pool1')

    model.add_node(Flatten(),
                   name='flatten', input='drop1')

model.add_node(Dense(512),
               name='dense', input='flatten')
model.add_node(Activation('relu'),
               name='relu4', input='dense')
model.add_node(Dropout(0.5),
               name='drop4', input='relu4')

model.add_node(Dense(nb_classes_fine),
               name='dense-f', input='drop4')
model.add_node(Activation('softmax'),
               name='soft-f', input='dense-f')
model.add_node(Dense(nb_classes_coarse),
               name='dense-c', input='drop4')
model.add_node(Activation('softmax'),
               name='soft-c', input='dense-c')

#TODO some other stuff here
model.add_output(name='output_fine', input='soft-f')
model.add_output(name='output_coarse', input='soft-c')


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss={'output_fine':'categorical_crossentropy','output_coarse':'categorical_crossentropy'}, optimizer=sgd)
history = model.fit({'input':X_train, 'output_fine':Y_train_fine, 'output_coarse':Y_train_coarse}, 
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    validation_data={'input':X_test, 'output_fine':Y_test_fine, 'output_coarse':Y_test_coarse})

model.save_weights('keras_cifar100_%s_weights.h5' % model_name)
json_string = model.to_json()
open('keras_cifar100_%s_architecture.json' % model_name, 'w').write(json_string)
pickle.dump(history, open('keras_cifar100_%s_history.p' % model_name,'r'))