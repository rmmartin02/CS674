import matplotlib.pyplot as plt
from math import log2
from pprint import pprint
import random
import sys
import os
import numpy as np
import pickle
import util

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
        cacheAlg = sys.argv[2]
        cacheSize = 1024 * int(sys.argv[3])
        ways = int(sys.argv[4])


    except:
        print('sim.py trace cacheAlg cacheSize(kB) #ofWays')

    lamb = .5 # 0=>LFU 1=>LRU

    blockSize = 64 // 8  # 64 bits, 8 bits per byte

    addressSize = 48 // 8  # bytes

    setSize = cacheSize // blockSize // ways

    blockBits = int(log2(blockSize))
    setBits = int(log2(setSize))

    hitTotal = 0
    missTotal = 0
    setCacheTotal = np.zeros(setSize, dtype=int)
    wayCacheTotal = np.zeros((setSize, ways), dtype=int)

    info = {}

    for file in files:

        print(file)

        with open('./Traces/{}'.format(file)) as f:
            trace = f.readlines()

        cache = np.zeros((setSize, ways, blockSize), dtype=int)

        if cacheAlg == 'LFU' or cacheAlg == 'MFU' or cacheAlg == 'LRU2':
            metaCache2 = np.zeros((setSize, ways), dtype=int)
        if cacheAlg == 'LRFU':
            metaCache = np.zeros((setSize, ways), dtype=float)
        else:
            metaCache = np.zeros((setSize, ways), dtype=int)
        setCache = np.zeros(setSize, dtype=int)
        wayCache = np.zeros((setSize, ways), dtype=int)

        print(cache, cache[0][0][0] == 0)
        print(setSize)

        print(len(trace))
        # trace = trace[:100]
        sets = [0] * len(trace)

        blockMask = int(''.join(['1'] * blockBits), 2)
        setMask = int(''.join(['1'] * setBits), 2)
        print(format(blockMask, "b"), format(setMask, "b"))

        hit = 0
        miss = 0

        for t in range(len(trace)):
            if t % 1000 == 0:
                print(t, len(trace), t / len(trace) * 100)
            a = int(trace[t].split(' ')[1])
            # print('Address',format(a, "b"))
            b = a & blockBits
            s = (a >> setBits) & setMask
            tag = (a >> setBits) >> blockBits

            setCache[s] += 1

            found = False
            for w in range(len(cache[s])):
                for b in cache[s][w]:
                    if tag == b:
                        hit += 1
                        found = True
                        if cacheAlg == 'LRU' or cacheAlg == 'MRU':
                            metaCache[s][w] = t
                        elif cacheAlg == 'LRU2':
                            # store second most recent timestamp
                            metaCache2[s][w] = metaCache[s][w]
                            metaCache[s][w] = t
                        elif cacheAlg == 'LFU' or cacheAlg == 'MFU':
                            metaCache[s][w] += 1
                        elif cacheAlg == 'FIFO' or cacheAlg == 'LIFO':
                            metaCache[s][w] = max(metaCache[s]) + 1
                        elif cacheAlg == 'PLRU':
                            metaCache[s][0] = (metaCache[s][0] + 1) % ways
                        elif cacheAlg == 'LRFU':
                            for i in range(len(metaCache[s])):
                                if cache[s][i][0] != 0 and i != w:
                                    metaCache[s][i] = (2 ** (-1 * lamb)) * metaCache[s][i]
                                else:
                                    metaCache[s][i] = 1 + (2 ** (-1 * lamb)) * metaCache[s][i]
                        break
            if not found:
                miss += 1
                w = -1
                for i in range(ways):
                    if cache[s][i][0] == 0:
                        w = i
                        break
                if w == -1:
                    if cacheAlg == 'R':
                        w = random.randint(0, len(cache[s]) - 1)
                    elif cacheAlg == 'LRU' or cacheAlg == 'LIFO' or cacheAlg == 'LRFU':
                        w = np.argmin(metaCache[s])
                    elif cacheAlg == 'LRU2':
                        w = np.argmin(metaCache2[s])
                    elif cacheAlg == 'MRU' or cacheAlg == 'FIFO':
                        w = np.argmax(metaCache[s])
                    elif cacheAlg == 'LFU':
                        # if there are multiple values decide by LRU
                        x = np.where(metaCache[s] == metaCache[s].min())
                        x = x[0]
                        w = x[0]
                        for y in range(len(x)):
                            if metaCache2[s][x[y]] < metaCache2[s][w]:
                                w = x[y]
                    elif cacheAlg == 'MFU':
                        # if there are multiple values decide by LRU
                        x = np.where(metaCache[s] == metaCache[s].max())
                        x = x[0]
                        w = x[0]
                        for y in range(len(x)):
                            if metaCache2[s][x[y]] < metaCache2[s][w]:
                                w = x[y]
                    elif cacheAlg == 'PLRU':
                        w = metaCache[s][0]

                if cacheAlg == 'LRU' or cacheAlg == 'MRU':
                    metaCache[s][w] = t
                elif cacheAlg == 'LRU2':
                    metaCache2[s][w] = t
                    metaCache[s][w] = t
                elif cacheAlg == 'LFU' or cacheAlg == 'MFU':
                    metaCache[s][w] = 1
                    metaCache2[s][w] = t
                elif cacheAlg == 'LIFO' or cacheAlg == 'FIFO':
                    metaCache[s][w] = max(metaCache[s]) + 1
                elif cacheAlg == 'PLRU':
                    metaCache[s][w] = 0
                elif cacheAlg == 'LRFU':
                    for i in range(len(metaCache[s])):
                        if i != w:
                            metaCache[s][i] = (2 ** (-1 * lamb)) * metaCache[s][i]
                        else:
                            metaCache[s][i] = 0

                wayCache[s][w] += 1
                for b in range(len(cache[s][w])):
                    cache[s][w][b] = tag


        hitTotal += hit
        missTotal += miss
        setCacheTotal = setCacheTotal + setCache
        wayCacheTotal = wayCacheTotal + wayCache

        print(metaCache)

        print(hit)
        print(miss)
        print(hit / (hit + miss))
        info[file] = {}
        info[file]['hits'] = hit
        info[file]['miss'] = miss
        info[file]['setCache'] = setCache
        info[file]['wayCache'] = wayCache
        info[file]['metaCache'] = metaCache
        if cacheAlg == 'LFU' or cacheAlg == 'MFU' or cacheAlg == 'LRU2':
            info[file]['metaCache2'] = metaCache2

    info['hitTotal'] = hitTotal
    info['missTotal'] = missTotal
    info['setCacheTotal'] = setCacheTotal
    info['wayCacheTotal'] = wayCacheTotal

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
