# analyze hyperparameter combinations
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns

fname = sys.argv[1]

trials = pickle.load(open(fname,'r'))

if 'hier2output' in fname:
    params = ['model_style', 'optimizer', 'batch_size']
else:
    params = ['dense_layer_size', 'num_conv1', 'num_conv2', 'optimizer', 'batch_size']

results = pd.DataFrame()

values = {x:np.array([]) for x in params}

"""
values = {'dense_layer_size':np.array([]),
          'num_conv1':np.array([]),
          'num_conv2':np.array([]),
          'optimizer':np.array([]),
          'batch_size':np.array([])
         }
"""
for r in trials.results:
    """
    results = results.append({
                    'accuracy':-r['loss'],
                    'dense_layer_size':int(r['params']['dense_layer_size']),
                    'num_conv1':int(r['params']['num_conv1']),
                    'num_conv2':int(r['params']['num_conv2']),
                    'optimizer':r['params']['optimizer'],
                    'batch_size':int(r['params']['batch_size'])
                   }, ignore_index=True)
    """
    data = {x:r['params'][x] for x in params}
    if 'hier2output' in fname:
        data['fine_acc'] = r['fine_acc']
        data['coarse_acc'] = r['coarse_acc']
        data['loss'] = r['loss']
    else:
        data['accuracy'] = -r['loss']
    results = results.append(data, ignore_index=True)
    for p in params:
        values[p] = np.append(values[p],r['params'][p])

# Looking at the data in different ways
extra_analysis=True
if extra_analysis:
    #results = results[(results['optimizer'] == 'adam') & (results['batch_size'] == 128)]
    results = results[(results['optimizer'] == 'adam') & (results['batch_size'] == 32)]

bar = []
for p in params:
    values[p] = np.unique(values[p])
    if 'hier2output' in fname:
        bar.append(sns.factorplot(p, 'coarse_acc', data=results, kind='bar', legend=True, x_order=values[p]))
        bar.append(sns.factorplot(p, 'fine_acc', data=results, kind='bar', legend=True, x_order=values[p]))
        bar.append(sns.factorplot(p, 'loss', data=results, kind='bar', legend=True, x_order=values[p]))
    else:
        bar.append(sns.factorplot(p, 'accuracy', data=results, kind='bar', legend=True, x_order=values[p]))


if 'hier2output' in fname:
    #sorted_data = results.sort('fine_acc')
    sorted_data = results.sort('loss')
else:
    sorted_data = results.sort('accuracy')

print(sorted_data)


plt.show()



