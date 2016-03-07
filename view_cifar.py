# View Images in the CIFAR Dataset
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sys
import time

num_train = 50000
num_test = 10000
PATH = '/home/bjkomer/semantic-network/data/cifar-100-python/'

# convert 1D numpy array to 3D image array
def convert_to_image(arr):
    return np.transpose(arr.reshape((32,32,3), order='F'), axes=(1,0,2))


# Accept input for which of the 60,000 images to show
if len(sys.argv) > 1:
    index = int(sys.argv[1])
else:
    index = -1

train_data = pickle.load(open(PATH + 'train', 'rb'))
test_data = pickle.load(open(PATH + 'test', 'rb'))

# Initialize Figure
im_fig = plt.figure(1, figsize=(1,1))
im_ax = im_fig.add_subplot(111)
im_ax.set_title("Cifar 100")
im_im = im_ax.imshow(np.zeros((32, 32, 3))) # Blank starting image
im_fig.show()
im_fig.canvas.draw()

if index == -1: # loop through all images
    for i in range(num_train):
        #im = train_data['data'][i].reshape((32,32,3), order='F')
        im = convert_to_image(train_data['data'][i])
        im_im.set_data(im)
        im_fig.canvas.draw()
        time.sleep(1)
