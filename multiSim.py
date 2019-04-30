'''
Attempt to multithread sim.py
'''

from math import log2
import random
import sys
import os
import multiprocessing
import numpy as np
import pickle
import time


# https://github.com/CMU-SAFARI/ramulator

# address size = 48 bits
# cache block size = 64 bits
# store address >> blockSize in cache

# loop to parallelize
def calculateSet(cacheAlg, addressArray, thrd, return_dict):
    cache = np.zeros((ways, blockSize), dtype=int)

    if cacheAlg == 'LFU' or cacheAlg == 'MFU' or cacheAlg == 'LRU2':
        metaCache2 = np.zeros(ways, dtype=int)
    if cacheAlg == 'LRFU':
        metaCache = np.zeros(ways, dtype=float)
    else:
        metaCache = np.zeros(ways, dtype=int)
    wayCache = np.zeros(ways, dtype=int)

    hit = 0
    miss = 0

    t = 0
    for a in addressArray:
        # if t % 1000 == 0:
        #    print(t, len(trace), t / len(trace) * 100)
        # print('Address',format(a, "b"))
        # b = a[1]
        tag = a[1]

        found = False
        for w in range(len(cache)):
            for b in cache[w]:
                if tag == b:
                    hit += 1
                    found = True
                    if cacheAlg == 'LRU' or cacheAlg == 'MRU':
                        metaCache[w] = t
                    elif cacheAlg == 'LRU2':
                        # store second most recent timestamp
                        metaCache2[w] = metaCache[w]
                        metaCache[w] = t
                    elif cacheAlg == 'LFU' or cacheAlg == 'MFU':
                        metaCache[w] += 1
                    elif cacheAlg == 'FIFO' or cacheAlg == 'LIFO':
                        metaCache[w] = max(metaCache) + 1
                    elif cacheAlg == 'PLRU':
                        metaCache[0] = (metaCache[0] + 1) % ways
                    elif cacheAlg == 'LRFU':
                        for i in range(len(metaCache)):
                            if cache[i][0] != 0 and i != w:
                                metaCache[i] = (2 ** (-1 * lamb)) * metaCache[i]
                            else:
                                metaCache[i] = 1 + (2 ** (-1 * lamb)) * metaCache[i]
                    break
        if not found:
            miss += 1
            w = -1
            for i in range(ways):
                if cache[i][0] == 0:
                    w = i
                    break
            if w == -1:
                if cacheAlg == 'R':
                    w = random.randint(0, len(cache) - 1)
                elif cacheAlg == 'LRU' or cacheAlg == 'LIFO' or cacheAlg == 'LRFU':
                    w = np.argmin(metaCache)
                elif cacheAlg == 'LRU2':
                    w = np.argmin(metaCache2)
                elif cacheAlg == 'MRU' or cacheAlg == 'FIFO':
                    w = np.argmax(metaCache)
                elif cacheAlg == 'LFU':
                    # if there are multiple values decide by LRU
                    x = np.where(metaCache == metaCache.min())
                    x = x[0]
                    w = x[0]
                    for y in range(len(x)):
                        if metaCache2[x[y]] < metaCache2[w]:
                            w = x[y]
                elif cacheAlg == 'MFU':
                    # if there are multiple values decide by LRU
                    x = np.where(metaCache == metaCache.max())
                    x = x[0]
                    w = x[0]
                    for y in range(len(x)):
                        if metaCache2[x[y]] < metaCache2[w]:
                            w = x[y]
                elif cacheAlg == 'PLRU':
                    w = metaCache[0]

            if cacheAlg == 'LRU' or cacheAlg == 'MRU':
                metaCache[w] = t
            elif cacheAlg == 'LRU2':
                metaCache2[w] = t
                metaCache[w] = t
            elif cacheAlg == 'LFU' or cacheAlg == 'MFU':
                metaCache[w] = 1
                metaCache2[w] = t
            elif cacheAlg == 'LIFO' or cacheAlg == 'FIFO':
                metaCache[w] = max(metaCache) + 1
            elif cacheAlg == 'PLRU':
                metaCache[w] = 0
            elif cacheAlg == 'LRFU':
                for i in range(len(metaCache)):
                    if i != w:
                        metaCache[i] = (2 ** (-1 * lamb)) * metaCache[i]
                    else:
                        metaCache[i] = 0

            wayCache[w] += 1
            for b in range(len(cache[w])):
                cache[w][b] = tag
            t += 1
    return_dict[thrd] = (hit, miss, wayCache)


if __name__ == "__main__":
    start = time.time()

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

    lamb = .5  # 0=>LFU 1=>LRU

    blockSize = 64 // 8  # 64 bits, 8 bits per byte

    addressSize = 48 // 8  # bytes

    setSize = cacheSize // blockSize // ways

    blockBits = int(log2(blockSize))
    setBits = int(log2(setSize))

    blockMask = int(''.join(['1'] * blockBits), 2)
    setMask = int(''.join(['1'] * setBits), 2)

    info = {}

    for file in files:

        print(file)

        with open('./Traces/{}'.format(file)) as f:
            trace = f.readlines()

        setDict = {}

        for t in range(len(trace)):
            if t % 1000 == 0:
                print(t, len(trace), t / len(trace) * 100)
            address = int(trace[t].split(' ')[1])
            # print('Address',format(a, "b"))
            setNum = (address >> setBits) & setMask
            block = address & blockBits
            tag = (address >> setBits) >> blockBits
            if setNum in setDict:
                setDict[setNum][0] += 1
                setDict[setNum].append((block, tag))
            else:
                setDict[setNum] = [1, setNum, (block, tag)]

        # sort so sets with most amount of accesses are done first and together
        # lengths are first entry of set

        setList = sorted(setDict.items(), key=lambda kv: kv[1][0])

        #(setList[0][1][2:])
        #sys.exit()

        numThreads = 1
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        for i in range(len(setList) // numThreads):
            jobs = []
            for j in range(numThreads):
                print(i*numThreads + j)
                p = multiprocessing.Process(target=calculateSet, args=(cacheAlg, setList[i * numThreads + j][1][2:], i * numThreads + j, return_dict))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

        #print(return_dict)

        hits = 0
        misses = 0
        wayCacheList = np.zeros((setSize, ways), dtype=int)
        setCacheList = np.zeros(setSize, dtype=int)
        for key in list(return_dict.keys()):
            hits += return_dict[key][0]
            misses += return_dict[key][1]
            wayCacheList[key] = return_dict[key][2]
            setCacheList[key] = hits+misses
        print(hits)

        info[file] = {}
        info[file]['hits'] = hits
        info[file]['miss'] = misses
        info[file]['setCache'] = wayCacheList
        info[file]['wayCache'] = setCacheList
        print(hits)

    with open('./Results/{}_{}KB_{}way.pickle'.format(sys.argv[2], sys.argv[3], sys.argv[4]), 'wb') as f:
        pickle.dump(info, f)

    end = time.time()
    print(end - start)


    # 834251
'''
import multiprocessing

def worker(procnum, return_dict):
    print str(procnum) + ' represent!'
    return_dict[procnum] = procnum


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print return_dict.values()

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
