from __future__ import print_function
gpu = 'gpu0'
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=%s,floatX=float32" % gpu
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, quniform
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import cPickle as pickle

# Open an IPython session if an exception is found
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

nb_epoch = 30 #NOTE: need to modify this elsewhere as well
nb_evals = 100

def data():

    nb_classes_fine = 100
    nb_classes_coarse = 20

    (X_train, y_train_fine), (X_test, y_test_fine) = cifar100.load_data(label_mode='fine')
    (_, y_train_coarse), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')

    # convert class vectors to binary class matrices
    Y_train_fine = np_utils.to_categorical(y_train_fine, nb_classes_fine)
    Y_train_coarse = np_utils.to_categorical(y_train_coarse, nb_classes_coarse)
    Y_test_fine = np_utils.to_categorical(y_test_fine, nb_classes_fine)
    Y_test_coarse = np_utils.to_categorical(y_test_coarse, nb_classes_coarse)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    #Y_train = np.concatenate((Y_train_coarse, Y_train_fine), axis=1)
    #Y_test = np.concatenate((Y_test_coarse, Y_test_fine), axis=1)

    return X_train, Y_train_fine, Y_train_coarse, X_test, Y_test_fine, Y_test_coarse

def model(X_train, Y_train_fine, Y_train_coarse, X_test, Y_test_fine, Y_test_coarse):

    nb_dim = 20
    img_rows, img_cols = 32, 32
    img_channels = 3

    #dense_layer_size = {{choice([256, 512, 1024])}}
    objective = 'mse'
    optimizer = {{choice(['rmsprop', 'adam', 'sgd'])}}
    batch_size = {{choice([32, 64, 128])}}
    #num_conv1 = int({{quniform(24, 64, 1)}})
    #num_conv2 = int({{quniform(32, 96, 1)}})
    model_style = {{choice(['original', 'wider', 'nodroporiginal', 'moredense', 'custom1', 'split', 'nodrop_split'])}}
    params = {#'dense_layer_size':dense_layer_size,
              'optimizer':optimizer,
              'batch_size':batch_size,
              #'num_conv1':num_conv1,
              #'num_conv2':num_conv2,
              'model_style':model_style
             }

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


    
    model.compile(loss={'output_fine':objective,'output_coarse':objective}, optimizer=optimizer, metrics=['accuracy'])

    history = model.fit({'input':X_train, 'output_fine':Y_train_fine,'output_coarse':Y_train_coarse}, batch_size=batch_size,
              nb_epoch=30, verbose=2,#show_accuracy=True,
              validation_data={'input':X_test, 'output_fine':Y_test_fine,'output_coarse':Y_test_coarse}, shuffle=True)

    #score, acc = model.evaluate({'input':X_train, 'output_fine':Y_train_fine,'output_coarse':Y_train_coarse}, verbose=0)
    loss, fine_loss, coarse_loss, fine_acc, coarse_acc = model.evaluate({'input':X_train, 'output_fine':Y_train_fine,'output_coarse':Y_train_coarse}, verbose=0)
    print('Test fine accuracy:', fine_acc)
    print('Test coarse accuracy:', coarse_acc)
    print('Combined loss', fine_loss + coarse_loss)
    #return {'loss': -acc, 'status': STATUS_OK, 'model':model}
    return {'loss': fine_loss + coarse_loss, 'status': STATUS_OK, 'params':params, 'fine_acc':fine_acc, 'coarse_acc':coarse_acc}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=nb_evals,
                                          trials=trials)

    X_train, Y_train_fine, Y_train_coarse, X_test, Y_test_fine, Y_test_coarse = data()
    #print("Evaluation of best performing model:")
    #print(best_model.evaluate(X_test, Y_test))
    
    print("saving results to: net_output/trials_hier2output_epoch%s_evals%s.p"%(nb_epoch, nb_evals))
    pickle.dump(trials, open('net_output/trials_hier2output_epoch%s_evals%s.p'%(nb_epoch, nb_evals),'w'))
