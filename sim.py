import matplotlib.pyplot as plt
from math import log2
from pprint import pprint
import random

addressSize = 64//8 #bytes
blocks = 64//addressSize
blockSize = blocks*addressSize
ways = 8
setSize = 256
cacheSize = blockSize * setSize * ways

if __name__ == "__main__":
    cache = []
    for s in range(setSize):
        cache.append([])
        for w in range(ways):
            cache[s].append([0]*int(blockSize/addressSize))

    pprint(cache)
    print(setSize)

    with open('Traces/433.milc') as f:
        trace = f.readlines()

    print(len(trace))
    #trace = trace[:100]
    sets = [0]*len(trace)

    blockBits = int(''.join(['1']*int(log2(blockSize))),2)
    setBits = int(''.join(['1']*int(log2(setSize))),2) << int(log2(blockSize))
    print(format(blockBits, "b"),format(setBits, "b"))

    for t in range(len(trace)):
        a = int(trace[t].split(' ')[1])
        #print('Address',format(a, "b"))
        b = a & blockBits
        s = a & setBits

        found = False
        for w in cache[s]:
            for b in w:
                if a == b:
                    found = True
                    break
        if not found:
            for w in cache[s]:
                r = random.randint(len(cache[s]))
                for b in range(len(cache[s][r])):
                    cache[s][r][b] = a &

        sets[t] = s

    plt.hist(sets,setSize)
    plt.show()