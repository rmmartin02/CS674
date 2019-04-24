'''
Cache simulator for only l1 cache
'''

from math import log2
import random
import sys
import os
import numpy as np
import pickle
from cache import Cache

# https://github.com/CMU-SAFARI/ramulator

# address size = 48 bits
# cache block size = 64 bits
# store address >> blockSize in cache

if __name__ == "__main__":

    try:
        if sys.argv[1] == 'all':
            files = os.listdir('./Traces/')
        else:
            files = [sys.argv[1]]
        cacheAlg = sys.argv[2]
        cacheSize = 1024 * int(sys.argv[3])
        ways = int(sys.argv[4])


    except:
        print('sim.py trace cacheAlg cacheSize(kB) #ofWays')

    cache = Cache(cacheAlg, cacheSize, ways)
    info = {}

    for file in files:
        print(file)

        cache.reset()

        with open('./Traces/{}'.format(file)) as f:
            trace = f.readlines()

        for t in range(len(trace)):
            if t % 1000 == 0:
                print(t, len(trace), t / len(trace) * 100)
            a = int(trace[t].split(' ')[1])

            found = cache.find(a)
            if not found:
                cache.load(a)

        info[file] = {}
        info[file]['hits'] = cache.hit
        info[file]['miss'] = cache.miss
        info[file]['setCache'] = cache.setCache
        info[file]['wayCache'] = cache.wayCache
        info[file]['metaCache'] = cache.metaCache
        if cacheAlg == 'LFU' or cacheAlg == 'MFU' or cacheAlg == 'LRU2':
            info[file]['metaCache2'] = cache.metaCache2

    with open('./Results/{}_{}KB_{}way.pickle'.format(sys.argv[2], sys.argv[3], sys.argv[4]), 'wb') as f:
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
