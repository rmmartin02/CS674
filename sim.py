import matplotlib.pyplot as plt
from math import log2
from pprint import pprint
import random

#address size = 48 bits
#cache block size = 64 bits
#store address >> blockSize in cache


cacheSize = 1024 * 32 #256 kBytes
ways = 8
blockSize = 64 // 8 #64 bits, 8 bits per byte

addressSize = 48//8 #bytes

setSize = cacheSize // blockSize // ways

blockBits = int(log2(blockSize))
setBits = int(log2(setSize))


if __name__ == "__main__":
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

    with open('Traces/433.milc') as f:
        trace = f.readlines()

    print(len(trace))
    #trace = trace[:100]
    sets = [0]*len(trace)

    blockMask = int(''.join(['1']*blockBits), 2)
    setMask = int(''.join(['1']*setBits), 2)
    print(format(blockMask, "b"), format(setMask, "b"))


    hit = 0
    miss = 0

    for t in range(len(trace)):
        a = int(trace[t].split(' ')[1])
        #print('Address',format(a, "b"))
        b = a & blockBits
        s = (a >> setBits) & setMask
        tag = (a >> setBits) >> blockBits

        print(a, tag, s, b)

        found = False
        for w in range(len(cache[s])):
            for b in cache[s][w]:
                if tag == b:
                    hit += 1
                    found = True
                    #frequency
                    metaCache[s][w] += 1
                    break
        if not found:
            miss += 1
            '''
            for w in cache[s]:
                r = random.randint(0, len(cache[s])-1)
                for b in range(len(cache[s][r])):
                    cache[s][r][b] = tag
            '''
            r = metaCache[s].index(min(metaCache[s]))
            for b in range(len(cache[s][r])):
                cache[s][r][b] = tag
            metaCache[s][r] = 0

        print(hit/(hit+miss))


