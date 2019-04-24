'''
A cache simulator that for a multilevel cache with l1 and l2 cache
'''


from math import log2
import random
import sys
import os
import numpy as np
import pickle
from cache import Cache

#https://github.com/CMU-SAFARI/ramulator

#address size = 48 bits
#cache block size = 64 bits
#store address >> blockSize in cache


if __name__ == "__main__":

    try:
        if sys.argv[1] == 'all':
            files = os.listdir('./Traces/')
        else:
            files = [sys.argv[1]]
        l1cacheAlg = sys.argv[2]
        l1cacheSize = 1024 * int(sys.argv[3])
        l1ways = int(sys.argv[4])
        l2cacheAlg = sys.argv[5]
        l2cacheSize = 1024 * int(sys.argv[6])
        l2ways = int(sys.argv[7])

    except IndexError:
        print('sim.py trace cacheAlg cacheSize(kB) #ofWays cacheAlg cacheSize(kB) #ofWays')

    l1cache = Cache(l1cacheAlg, l1cacheSize, l1ways)
    l2cache = Cache(l2cacheAlg, l2cacheSize, l2ways)
    info = {}

    for file in files:
        print(file)

        l1cache.reset()
        l2cache.reset()

        with open('./Traces/{}'.format(file)) as f:
            trace = f.readlines()

        for t in range(len(trace)):
            if t % 1000 == 0:
                print(t, len(trace), t / len(trace) * 100)
            a = int(trace[t].split(' ')[1])

            found = l1cache.find(a)
            if not found:
                found = l2cache.find(a)
                if not found:
                    l2cache.load(a)
                    # could make them not consistent here?
                l1cache.load(a)

        print(l1cache.hit / (l1cache.hit+l1cache.miss))
        print(l2cache.hit / (l2cache.hit+l2cache.miss))

        info[file] = {}
        info[file]['l1'] = {}
        info[file]['l1']['hits'] = l1cache.hit
        info[file]['l1']['miss'] = l1cache.miss
        info[file]['l1']['setCache'] = l1cache.setCache
        info[file]['l1']['wayCache'] = l1cache.wayCache
        info[file]['l1']['metaCache'] = l1cache.metaCache
        if l1cacheAlg == 'LFU' or l1cacheAlg == 'MFU' or l1cacheAlg == 'LRU2':
            info[file]['l1']['metaCache2'] = l1cache.metaCache2

        info[file]['l2'] = {}
        info[file]['l2']['hits'] = l2cache.hit
        info[file]['l2']['miss'] = l2cache.miss
        info[file]['l2']['setCache'] = l2cache.setCache
        info[file]['l2']['wayCache'] = l2cache.wayCache
        info[file]['l2']['metaCache'] = l2cache.metaCache
        if l2cacheAlg == 'LFU' or l2cacheAlg == 'MFU' or l2cacheAlg == 'LRU2':
            info[file]['l2']['metaCache2'] = l2cache.metaCache2

    with open('./Results/{}_{}KB_{}way_{}_{}KB_{}way.pickle'.format(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]), 'wb') as f:
        pickle.dump(info, f)

    '''

    plt.bar(range(setSize), setCache)

    plt.show()


    print(len(wayCache), len(wayCache[0]))

    for i in range(setSize // 64):

        plt.subplot(1, setSize // 64, i + 1)

        plt.imshow(wayCache[i * 64:(i + 1) * 64], cmap='Reds', interpolation='nearest')

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

    plt.show()

    '''
