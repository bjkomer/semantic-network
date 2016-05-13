# analyze hyperparameter combinations
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns

fname = sys.argv[1]

trials = pickle.load(open(fname,'r'))
params = ['dense_layer_size', 'num_conv1', 'num_conv2', 'optimizer', 'batch_size']

results = pd.DataFrame()

values = {'dense_layer_size':np.array([]),
          'num_conv1':np.array([]),
          'num_conv2':np.array([]),
          'optimizer':np.array([]),
          'batch_size':np.array([])
         }

for r in trials.results:

    results = results.append({
                    'accuracy':-r['loss'],
                    'dense_layer_size':int(r['params']['dense_layer_size']),
                    'num_conv1':int(r['params']['num_conv1']),
                    'num_conv2':int(r['params']['num_conv2']),
                    'optimizer':r['params']['optimizer'],
                    'batch_size':int(r['params']['batch_size'])
                   }, ignore_index=True)
    for p in params:
        values[p] = np.append(values[p],r['params'][p])

bar = []
for p in params:
    values[p] = np.unique(values[p])
    #bar.append(sns.barplot(y=p, x='accuracy', data=results, kind='bar', legend=True))
    bar.append(sns.factorplot(p, 'accuracy', data=results, kind='bar', legend=True, x_order=values[p]))

plt.show()



