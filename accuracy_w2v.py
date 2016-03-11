# Get the accuracy of a trained w2v net
import numpy as np
import getpass
import sys
import h5py
#import w2v_classifier

# Check the username, so the same code can work on all of our computers
user = getpass.getuser()
if user == 'ctnuser':
    caffe_root = '/home/ctnuser/bjkomer/caffe/'
    root = '/home/ctnuser/bjkomer/'
elif user == 'bjkomer':
    caffe_root = '/home/bjkomer/caffe/'
    root = '/home/bjkomer/'

sys.path.insert(0, caffe_root + 'python')

import caffe

if user == 'ctnuser':
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

# load images and output labels
DIM = 200
NUM_TRAIN = 50000
NUM_TEST = 10000
data_prefix = 'data/cifar_100_caffe_hdf5/'
fnametrain = data_prefix + 'train.h5'
fnametest = data_prefix + 'test.h5'
fnametrain_label = data_prefix + 'train_w2v_label.h5'
fnametest_label = data_prefix + 'test_w2v_label.h5'

# Training
ftrain = h5py.File(fnametrain, 'r')
ftrain_w2v = h5py.File(fnametrain_label, 'r')
train_label = np.zeros((NUM_TRAIN)) # integer labels
train_image = np.zeros((NUM_TRAIN, 3, 32, 32)) # actual images
train_w2v_label = np.zeros((NUM_TRAIN, DIM)) # word2vec labels

for i, label in enumerate(ftrain['label_fine']):
    train_label[i] = label
for i, image in enumerate(ftrain['data']):
    train_image[i,:] = image
for i, w2v_label in enumerate(ftrain_w2v['label_w2v']):
    train_w2v_label[i,:] = w2v_label

# Testing
ftest = h5py.File(fnametest, 'r')
ftest_w2v = h5py.File(fnametest_label, 'r')
test_label = np.zeros((NUM_TEST))
test_image = np.zeros((NUM_TEST, 3, 32, 32))
test_w2v_label = np.zeros((NUM_TEST, DIM))

for i, label in enumerate(ftest['label_fine']):
    test_label[i] = label
for i, image in enumerate(ftest['data']):
    test_image[i,:] = image
for i, w2v_label in enumerate(ftest_w2v['label_w2v']):
    test_w2v_label[i,:] = w2v_label

# Get all of the unique label vectors
b = np.ascontiguousarray(test_w2v_label).view(np.dtype((np.void, test_w2v_label.dtype.itemsize * test_w2v_label.shape[1])))
_, idx = np.unique(b, return_index=True)
all_vectors = test_w2v_label[idx]

TOL = 0.0001
# returns 1 if output is closer to vector than to all other vectors
def check_match(output, vector):
    best_diff = None
    best_vector = None
    for v in all_vectors:
        diff = np.linalg.norm(output - v)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_vector = v
    # check that the vector is the same within some numerical tolerance
    if np.linalg.norm(vector - best_vector) < TOL:
        return 1
    else:
        return 0

########################
# Set up caffe network #
########################
net = caffe.Net(root + 'net/cifar100_w2v_deploy.prototxt',
                root + 'net_output/cifar100_w2v_snapshot_iter_150000.caffemodel',
                caffe.TEST)

"""
# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
"""

batch_size = 10
output_layer = 'ip_c'
net.blobs['data'].reshape(batch_size,3,32,32) # Cifar100 uses 32x32


# run caffe network on test images
test_result = np.zeros(NUM_TEST)
for batch in range(int(NUM_TEST / batch_size)):
    net.blobs['data'].data[...] = test_image[batch*batch_size:(batch+1)*batch_size,:,:,:]
    print("Running Testing Batch %i of %i" % (batch, int(NUM_TEST / batch_size)))
    out = net.forward()

    for bi in range(batch_size):

      output = net.blobs[layer].data[bi]
      test_result[batch*batch_size+bi] = check_match(output, test_w2v_label[batch*batch_size+bi,:])

      # get closest label vector to the output vector
      #label = w2v_classifier.classify(output)

test_accuracy = np.mean(test_result)
print("Test Accuracy: %f" % test_accuracy)