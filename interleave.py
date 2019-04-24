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
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        cacheAlg = sys.argv[3]
        cacheSize = 1024 * int(sys.argv[4])
        ways = int(sys.argv[5])


    except:
        print('sim.py trace cacheAlg cacheSize(kB) #ofWays')

    cache = Cache(cacheAlg, cacheSize, ways)
    info = {}

    cache.reset()

    with open('./Traces/{}'.format(file1)) as f:
        trace1 = f.readlines()
    trace1 = [int(a.split(' ')[-1].rstrip()) for a in trace1]

    with open('./Traces/{}'.format(file2)) as f:
        trace2 = f.readlines()
    trace2 = [int(a.split(' ')[-1].rstrip()) for a in trace2]

    if len(trace1)>len(trace2):
        trace1 = trace1[:len(trace2)]
    else:
        trace2 = trace2[:len(trace1)]
    print(len(trace1), len(trace2))

    trace = [0] * (len(trace1)*2)
    for i in range(0, len(trace), 2):
        trace[i] = trace1[i//2]
        trace[i+1] = trace2[i//2]

    for t in range(len(trace)):
        if t % 1000 == 0:
            print(t, len(trace), t / len(trace) * 100)
        a = trace[t]

        found = cache.find(a)
        if not found:
            cache.load(a)

    info = {}
    info['hits'] = cache.hit
    info['miss'] = cache.miss
    info['setCache'] = cache.setCache
    info['wayCache'] = cache.wayCache
    info['metaCache'] = cache.metaCache
    if cacheAlg == 'LFU' or cacheAlg == 'MFU' or cacheAlg == 'LRU2':
        info['metaCache2'] = cache.metaCache2

    with open('./Results/{}_{}_{}_{}KB_{}way.pickle'.format(sys.argv[1],sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]), 'wb') as f:
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
