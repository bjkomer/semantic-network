import word2vec

word2vec.word2phrase('/home/bjkomer/word2vec/text8',
                     '/home/bjkomer/semantic-network/data/text8-phrases', verbose=True)

word2vec.word2vec('/home/bjkomer/semantic-network/data/text8-phrases',
                  '/home/bjkomer/semantic-network/data/text8.bin', size=100,
                  verbose=True)

word2vec.word2clusters('/home/bjkomer/word2vec/text8',
                       '/home/bjkomer/semantic-network/data/text8-clusters.txt', 100,
                       verbose=True)
