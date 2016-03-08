import word2vec
import numpy as np

classes = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
           'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
           'bottles', 'bowls', 'cans', 'cups', 'plates',
           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
           'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
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
           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
           'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
          ]
# renamed slightly to match existing words in word2vec corpus
# NOTE: the biggest difference is probably lawn-mower being changed to cutter
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

#model = word2vec.load('/home/bjkomer/semantic-network/data/word2vec/GoogleNews-vectors-negative300.bin',
#                      encoding='ISO-8859-1'
#                     )
model = word2vec.load('/home/bjkomer/word2vec/vectors.bin')
#model = word2vec.load('/home/bjkomer/semantic-network/data/text8.bin')

for c in classes:
    print(c)
    print(model[c][:5])
    print(model[c].shape)
"""
print(model)
print(dir(model))

print(model['cat'][:10])
print(model['cat'].shape)
"""
