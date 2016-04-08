import word2vec
import sys

if len(sys.argv) == 2:
    dim = int(sys.argv[1])
else:
    dim = 100

word2vec.word2phrase('/home/bjkomer/word2vec/text8',
                     '/home/bjkomer/semantic-network/data/text8-phrases', verbose=True)

word2vec.word2vec('/home/bjkomer/semantic-network/data/text8-phrases',
                  '/home/bjkomer/semantic-network/data/text8-%s.bin'%dim, size=dim,
                  verbose=True)

word2vec.word2clusters('/home/bjkomer/word2vec/text8',
                       '/home/bjkomer/semantic-network/data/text8-%s-clusters.txt'%dim, dim,
                       verbose=True)
