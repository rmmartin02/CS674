import matplotlib.pyplot as plt
from math import log2
from pprint import pprint
import random
import sys

#https://github.com/CMU-SAFARI/ramulator

#address size = 48 bits
#cache block size = 64 bits
#store address >> blockSize in cache


if __name__ == "__main__":
    try:
        with open(sys.argv[1]) as f:
            trace = f.readlines()
        cacheAlg = sys.argv[2]
        cacheSize = 1024 * int(sys.argv[3])
        ways = int(sys.argv[4])
    except:
        print('sim.py trace cacheAlg cacheSize(kB) #ofWays')

    blockSize = 64 // 8  # 64 bits, 8 bits per byte

    addressSize = 48 // 8  # bytes

    setSize = cacheSize // blockSize // ways

    blockBits = int(log2(blockSize))
    setBits = int(log2(setSize))

    cache = []
    metaCache = [] #holds metadata such as frequency
    for s in range(setSize):
        cache.append([])
        metaCache.append([])
        for w in range(ways):
            cache[s].append([0]*blockSize)
            metaCache[s].append(0)

    pprint(cache)
    print(setSize)

    print(len(trace))
    #trace = trace[:100]
    sets = [0]*len(trace)

    blockMask = int(''.join(['1']*blockBits), 2)
    setMask = int(''.join(['1']*setBits), 2)
    print(format(blockMask, "b"), format(setMask, "b"))


    hit = 0
    miss = 0

    setDict = {}

    for t in range(len(trace)):
        print(t, len(trace), t/len(trace)*100)
        a = int(trace[t].split(' ')[1])
        #print('Address',format(a, "b"))
        b = a & blockBits
        s = (a >> setBits) & setMask
        tag = (a >> setBits) >> blockBits

        if s in setDict:
            setDict[s]+=1
        else:
            setDict[s] = 1
        #print(a, tag, s, b)

        found = False
        for w in range(len(cache[s])):
            for b in cache[s][w]:
                if tag == b:
                    hit += 1
                    found = True
                    if cacheAlg == 'LRU' or cacheAlg == 'MRU':
                        metaCache[s][w] += 1
                    if cacheAlg == 'FIFO' or cacheAlg == 'LIFO':
                        metaCache[s][w] = max(metaCache[s])+1
                    if cacheAlg == 'PLRU':
                        metaCache[s][0] = (metaCache[s][0]+1)%ways
                    break
        if not found:
            miss += 1
            r = 0
            if cacheAlg == 'R':
                for w in cache[s]:
                    r = random.randint(0, len(cache[s])-1)
            elif cacheAlg == 'LRU':
                r = metaCache[s].index(min(metaCache[s]))
                metaCache[s][r] = 0
            elif cacheAlg == 'MRU':
                r = metaCache[s].index(max(metaCache[s]))
                metaCache[s][r] = 0
            elif cacheAlg == 'LIFO':
                r = metaCache[s].index(max(metaCache[s]))
            elif cacheAlg == 'FIFO':
                r = metaCache[s].index(min(metaCache[s]))
            elif cacheAlg == 'PLRU':
                r = metaCache[s][0]
            for b in range(len(cache[s][r])):
                cache[s][r][b] = tag

    setList = [(k, setDict[k]) for k in sorted(setDict, key=setDict.get, reverse=True)]
    plt.bar([a[0] for a in setList], [a[1] for a in setList])
    plt.show()
    print(setList)
    print(hit)
    print(miss)
    print(hit/(hit+miss))


