# plots training vs. validation loss for each epoch
# used to determine if more training is needed or it is overfitting
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 2:
    fname = sys.argv[1]

    data = pickle.load(open(fname, 'r'))



    plt.figure()
    plt.title(fname)
    plt.plot(data['loss'], label='loss')
    plt.plot(data['val_loss'], label='val_loss')
    plt.legend()
elif len(sys.argv) == 3:
    fname1 = sys.argv[1]
    fname2 = sys.argv[2]

    data1 = pickle.load(open(fname1, 'r'))
    data2 = pickle.load(open(fname2, 'r'))



    plt.figure()
    plt.title(fname1 + " and " + fname2)
    plt.plot(data1['loss'], label='loss1')
    plt.plot(data1['val_loss'], label='val_loss1')
    plt.plot(data2['loss'], label='loss2')
    plt.plot(data2['val_loss'], label='val_loss2')
    plt.legend()
elif len(sys.argv) > 3:
    #loss
    plt.figure()
    for fname in sys.argv[1:]:
        data = pickle.load(open(fname, 'r'))
        plt.plot(data['loss'], label='loss_'+fname)

    plt.legend()
    
    # val_loss
    plt.figure()
    for fname in sys.argv[1:]:
        data = pickle.load(open(fname, 'r'))
        plt.plot(data['val_loss'], label='val_loss_'+fname)

    plt.legend()

plt.show()
