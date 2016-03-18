# Common utilities for neural networks
import numpy as np
import h5py

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
