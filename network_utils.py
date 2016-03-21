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
