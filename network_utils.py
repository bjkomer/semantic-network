# Common utilities for neural networks
import numpy as np
import h5py
import word2vec
import getpass

user = getpass.getuser()
if user == 'ctnuser':
    root = '/home/ctnuser/bjkomer/'
elif user == 'bjkomer':
    root = '/home/bjkomer/'
else:
    root = '/home/'+user+'/'

classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
           'fish', 'flounder', 'ray', 'shark', 'trout',
           'orchids', 'poppies', 'roses', 'sunflower', 'tulips',
           'bottles', 'bowls', 'cans', 'cups', 'plates',
           'apples', 'mushrooms', 'oranges', 'pears', 'peppers',
           'clock', 'keyboard', 'lamp', 'telephone', 'television',
           'bed', 'chair', 'couch', 'table', 'wardrobe',
           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
           'bear', 'leopard', 'lion', 'tiger', 'wolf',
           'bridge', 'castle', 'house', 'road', 'skyscraper',
           'cloud', 'forest', 'mountain', 'plain', 'sea',
           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
           'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
           'crab', 'lobster', 'snail', 'spider', 'worm',
           'baby', 'boy', 'girl', 'man', 'woman',
           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
           'maple', 'oak', 'palm', 'pine', 'willow',
           'bicycle', 'bus', 'motorcycle', 'truck', 'train',
           'cutter', 'rocket', 'streetcar', 'tank', 'tractor'
          ]

# Computes the accuracy of a prediction
def accuracy(prediction, actual):
    num = len(actual)
    correct = 0.0
    for i in range(num):
        if np.array_equal(prediction[i], actual[i]):
            correct += 1
    return correct / num


# Get all of the unique label vectors
data_prefix = 'data/cifar_100_caffe_hdf5/'
#fnametest_label = data_prefix + 'test_w2v_label.h5'
DIM = 200
NUM_CLASSES = 100
"""
NUM_TEST = 10000
print("Finding unique label vector set")
ftest_w2v = h5py.File(fnametest_label, 'r')
test_w2v_label = np.zeros((NUM_TEST, DIM))
test_w2v_label = ftest_w2v['label_w2v'][()]
b = np.ascontiguousarray(test_w2v_label).view(np.dtype((np.void, test_w2v_label.dtype.itemsize * test_w2v_label.shape[1])))
_, idx = np.unique(b, return_index=True)
all_vectors = test_w2v_label[idx]
"""
fnamelist_label = data_prefix + 'ordered_w2v_labels_200dim.h5'
flist_w2v = h5py.File(fnamelist_label, 'r')
all_vectors = np.zeros((NUM_CLASSES, DIM))
all_vectors = flist_w2v['label_w2v'][()]
TOL = 0.0001

def accuracy_w2v(prediction, actual):
    num = len(actual)
    correct = 0.0
    pred_class = np.zeros((num, 100)) # which class number was being predicted
    for i in range(num):
        best_diff = None
        best_vector = None
        best_class = None # index of the best class
        #for j, v in enumerate(all_vectors[::-1]):
        for j, v in enumerate(all_vectors):
            diff = np.linalg.norm(prediction[i] - v)
            if (best_diff is None) or (diff < best_diff):
                best_diff = diff
                best_vector = v
                best_class = j
        # check that the vector is the same within some numerical tolerance
        pred_class[i,best_class] += 1
        if np.linalg.norm(actual[i] - best_vector) < TOL:
            correct += 1
    return correct / num, pred_class

def load_custom_weights(model, filepath, layer_indices=[0,1,2,3,4,5,6,7,8,9,10,11,14,15]):
    f = h5py.File(filepath, mode='r')
    g = f['graph']

    weights = [g['param_{}'.format(p)] for p in layer_indices]
    model.set_weights(weights)
    f.close()

def get_w2v_labels(y_original, dim=200):
    y_new = np.zeros((y_original.shape[0], dim))
    if dim == 200:
        model = word2vec.load(root + 'word2vec/vectors.bin')
    else:
        raise NotImplementedError
    for i, label in enumerate(y_original):
        y_new[i,:] = model[classes[label]]

    return y_new

def clean_hierarchy_vec(vectors):
    # first 20 elements are coarse label
    # next 100 elements are fine label
    num = len(vectors)
    output = np.zeros((num, 120))
    for i,v in enumerate(vectors):
        ci = np.argmax(v[:20])
        fi = np.argmax(v[20:])
        output[i,ci] = 1
        output[i,fi] = 1
    return output

# Computes the accuracy of a prediction for hierarchical vectors
def accuracy_hierarchy(prediction, actual):
    num = len(actual)
    correct = 0.0
    correct_coarse = 0.0
    correct_fine = 0.0
    for i in range(num):
        if np.array_equal(prediction[i], actual[i]):
            correct += 1
        if np.array_equal(prediction[i,:20], actual[i,:20]):
            correct_coarse += 1
        if np.array_equal(prediction[i,20:], actual[i,20:]):
            correct_fine += 1
    return correct / num, correct_coarse / num, correct_fine / num
