# Standard labels for Cifar100 using Keras
# Matching the network structure of the Caffe network for Cifar100
from __future__ import print_function
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import cPickle as pickle
from network_utils import accuracy

# Open an IPython session if an exception is found
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

batch_size = 32
nb_classes_fine = 100
nb_classes_coarse = 20
nb_epoch = 50#200
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

# some preprocessing stuff
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

######################
# Beginning of Model #
######################
model = Graph()

model.add_input(name='input', input_shape=(img_channels, img_rows, img_cols))

model_name='matching_preproc'

model.add_node(Convolution2D(64, 4, 4, border_mode='same', init='glorot_uniform',
                        input_shape=(img_channels, img_rows, img_cols)),
               name='conv1', input='input')
model.add_node(Convolution2D(42, 1, 1, init='glorot_uniform'),
               name='cccp1a', input='conv1')
model.add_node(Activation('relu'),
               name='relu1a', input='cccp1a')
model.add_node(Convolution2D(32, 1, 1, init='glorot_uniform'),
               name='cccp1b', input='relu1a')
model.add_node(MaxPooling2D(pool_size=(3, 3),strides=(2,2)),
               name='pool1', input='cccp1b')
model.add_node(Dropout(0.5),
               name='drop1', input='pool1')
model.add_node(Activation('relu'),
               name='relu1b', input='drop1')
model.add_node(Convolution2D(42, 4, 4),
               name='conv2', input='relu1b')
model.add_node(MaxPooling2D(pool_size=(3, 3),strides=(2,2)),
               name='pool2', input='conv2')
model.add_node(Dropout(0.5),
               name='drop2', input='pool2')
model.add_node(Activation('relu'),
               name='relu2', input='drop2')
model.add_node(Convolution2D(64, 2, 2, init='glorot_uniform'),
               name='conv3', input='relu2')
model.add_node(AveragePooling2D(pool_size=(2, 2), strides=(2,2)),
               name='pool3', input='conv3')
model.add_node(Activation('relu'),
               name='relu3', input='pool3')
model.add_node(Flatten(),
               name='flat', input='relu3')
#inner product layer
model.add_node(Dense(768, init='glorot_uniform'),
               name='ip1', input='flat')
#sigmoid layer
model.add_node(Activation('sigmoid'),
               name='sig1', input='ip1')

#inner product c
model.add_node(Dense(20, init='glorot_uniform'),
               name='ip_c', input='ip1')
#accuracy c
#loss c
model.add_node(Activation('softmax'),
               name='soft_c', input='ip_c')
#ip_f
model.add_node(Dense(100, init='glorot_uniform'),
               name='ip_f', input='ip1')
#accuracy_f
#loss_f
model.add_node(Activation('softmax'),
               name='soft_f', input='ip_f')

#TODO some other stuff here
model.add_output(name='output_fine', input='soft_f')
model.add_output(name='output_coarse', input='soft_c')


training = True # if the network should train, or just load the weights from elsewhere
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
sgd = SGD(lr=0.0006, decay=0.001, momentum=0.0, nesterov=True)
#model.compile(loss={'output_fine':'categorical_crossentropy','output_coarse':'categorical_crossentropy'}, optimizer=sgd)
model.compile(loss={'output_fine':'mean_squared_error','output_coarse':'mean_squared_error'}, optimizer=sgd)
if training:
    if not data_augmentation:
        history = model.fit({'input':X_train, 'output_fine':Y_train_fine, 'output_coarse':Y_train_coarse}, 
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            validation_data={'input':X_test, 'output_fine':Y_test_fine, 'output_coarse':Y_test_coarse})
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

    model.save_weights('net_output/keras_cifar100_%s_weights.h5' % model_name)
    json_string = model.to_json()
    open('net_output/keras_cifar100_%s_architecture.json' % model_name, 'w').write(json_string)
    pickle.dump(history.history, open('net_output/keras_cifar100_%s_history.p' % model_name,'w'))
    print("saving to: keras_cifar100_%s" % model_name)
else:
    model.load_weights('net_output/keras_cifar100_%s_weights.h5' % model_name)
    Y_predict_test = model.predict({'input':X_test}, batch_size=batch_size, verbose=1)
    Y_predict_train = model.predict({'input':X_train}, batch_size=batch_size, verbose=1)
    
    Y_predict_test_fine = Y_predict_test['output_fine']
    Y_predict_test_coarse = Y_predict_test['output_coarse']
    test_accuracy_fine = accuracy(Y_predict_test_fine, Y_test_fine)
    test_accuracy_coarse = accuracy(Y_predict_test_coarse, Y_test_coarse)
    print("Fine test accuracy: %f" % test_accuracy_fine)
    print("Coarse test accuracy: %f" % test_accuracy_coarse)
    
    Y_predict_train_fine = Y_predict_train['output_fine']
    Y_predict_train_coarse = Y_predict_train['output_coarse']
    train_accuracy_fine = accuracy(Y_predict_train_fine, Y_train_fine)
    train_accuracy_coarse = accuracy(Y_predict_train_coarse, Y_train_coarse)
    print("Fine train accuracy: %f" % train_accuracy_fine)
    print("Coarse train accuracy: %f" % train_accuracy_coarse)
