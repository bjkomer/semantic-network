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
NUM_COARSE_CLASSES = 20
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
#fnamelist_label = data_prefix + 'ordered_w2v_labels_%sdim.h5'%DIM
#flist_w2v = h5py.File(fnamelist_label, 'r')
#all_vectors = np.zeros((NUM_CLASSES, DIM))
#all_vectors = flist_w2v['label_w2v'][()]
TOL = 0.0001

def accuracy_w2v(prediction, actual, dim):
    fnamelist_label = data_prefix + 'ordered_w2v_labels_%sdim.h5'%dim
    flist_w2v = h5py.File(fnamelist_label, 'r')
    all_vectors = np.zeros((NUM_CLASSES, DIM))
    all_vectors = flist_w2v['label_w2v'][()]
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

# Two ways of measuring this. One is if it is closest to any of the fine vectors in the coarse category, it is considered correct
# The other is if it is closest to the mean vector of the fine vectors (i.e. coarse vector) then it is correct
def accuracy_w2v_coarse(prediction, actual, dim):
    fnamelist_label = data_prefix + 'ordered_w2v_labels_%sdim.h5'%dim
    flist_w2v = h5py.File(fnamelist_label, 'r')
    all_vectors = np.zeros((NUM_CLASSES, dim))
    all_vectors = flist_w2v['label_w2v'][()]
    all_coarse_vectors = np.zeros((NUM_COARSE_CLASSES, dim))
    
    # Average the five fine vectors into one coarse vector
    for i, v in enumerate(all_vectors):
        all_coarse_vectors[i/5,:] += all_vectors[i,:] / 5.

    num = len(actual)
    correct = 0.0
    correct_coarse = 0.0
    correct_coarse_v2 = 0.0
    pred_class = np.zeros((num, 100)) # which class number was being predicted
    pred_coarse_class = np.zeros((num, 20)) # which coarse class number was being predicted
    pred_coarse_class_v2 = np.zeros((num, 20)) # which coarse class number was being predicted in version 2 method
    for i in range(num):
        best_diff = None
        best_vector = None
        best_class = None # index of the best class
        best_coarse_class = None # index of the best coarse class
        
        best_diff_v2 = None
        best_vector_v2 = None
        best_coarse_class_v2 = None # index of the best coarse class for version two of accuracy metric
        actual_class_v2 = None
        
        best_diff_ac_v2 = None # used for finding the 'actual' correct vector for coarse classification version 2

        for j, v in enumerate(all_vectors):
            diff = np.linalg.norm(prediction[i] - v)
            if (best_diff is None) or (diff < best_diff):
                best_diff = diff
                best_vector = v
                best_class = j
                best_coarse_class = j / 5
            
            diff = np.linalg.norm(actual[i] - v)
            if (best_diff_ac_v2 is None) or (diff < best_diff_ac_v2):
                best_diff_ac_v2 = diff
                best_vector_ac_v2 = v
                actual_class_v2 = j / 5
        
        for j, v in enumerate(all_coarse_vectors):
            diff = np.linalg.norm(prediction[i] - v)
            if (best_diff_v2 is None) or (diff < best_diff_v2):
                best_diff_v2 = diff
                best_vector_v2 = v
                best_coarse_class_v2 = j
            
        
        # check that the vector is the same within some numerical tolerance
        pred_class[i,best_class] += 1
        pred_coarse_class[i,best_coarse_class] += 1
        pred_coarse_class_v2[i,best_coarse_class_v2] += 1
        if np.linalg.norm(actual[i] - best_vector) < TOL:
            correct += 1
        
        if best_coarse_class_v2 == actual_class_v2:
            correct_coarse_v2 += 1

        # figure out coarse correctness
        # need to first get the set of fine vectors for the coarse class
        coarse_vectors = all_vectors[best_coarse_class*5:(best_coarse_class+1)*5]
        for v in coarse_vectors:
            if np.linalg.norm(actual[i] - v) < TOL:
                correct_coarse += 1
                break

    return correct / num, pred_class, correct_coarse / num, pred_coarse_class, correct_coarse_v2 / num, pred_coarse_class_v2

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
    elif dim in [100,50,25,10]:
        model = word2vec.load(root + 'semantic-network/data/text8-%s.bin'%dim)
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
        fi = np.argmax(v[20:])+20
        output[i,ci] = 1
        output[i,fi] = 1
    return output

def clean_vec(vectors):
    # works for either coarse or fine labels
    num = len(vectors)
    output = np.zeros((num, len(vectors[0])))
    for i,v in enumerate(vectors):
        li = np.argmax(v)
        output[i,li] = 1
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
