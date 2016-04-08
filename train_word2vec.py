import word2vec
import sys
import getpass

user = getpass.getuser()
if user == 'ctnuser':
    root = '/home/ctnuser/bjkomer/'
elif user == 'bjkomer':
    root = '/home/bjkomer/'

if len(sys.argv) == 2:
    dim = int(sys.argv[1])
else:
    dim = 100

word2vec.word2phrase(root + 'word2vec/text8',
                     root + 'semantic-network/data/text8-phrases', verbose=True)

word2vec.word2vec(root + 'semantic-network/data/text8-phrases',
                  root + 'semantic-network/data/text8-%s.bin'%dim, size=dim,
                  verbose=True)

word2vec.word2clusters(root + 'word2vec/text8',
                       root + 'semantic-network/data/text8-%s-clusters.txt'%dim, dim,
                       verbose=True)
