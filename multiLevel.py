'''
A cache simulator that for a multilevel cache with l1 and l2 cache
'''


from math import log2
import random
import sys
import os
import numpy as np
import pickle

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


    except:
        print('sim.py trace l1cacheAlg l1cacheSize(kB) #ofWays')

    lamb = .5 # 0=>LFU 1=>LRU

    blockSize = 64 // 8  # 64 bits, 8 bits per byte

    addressSize = 48 // 8  # bytes

    l1setSize = l1cacheSize // blockSize // l1ways
    l2setSize = l2cacheSize // blockSize // l2ways

    blockBits = int(log2(blockSize))
    l1setBits = int(log2(l1setSize))
    l2setBits = int(log2(l2setSize))

    hitTotal = 0
    missTotal = 0
    setl1cacheTotal = np.zeros(l1setSize, dtype=int)
    wayl1cacheTotal = np.zeros((l1setSize, l1ways), dtype=int)

    info = {}

    for file in files:

        print(file)

        with open('./Traces/{}'.format(file)) as f:
            trace = f.readlines()

        # setup l1 cache
        l1cache = np.zeros((l1setSize, l1ways, blockSize), dtype=int)

        if l1cacheAlg == 'LFU' or l1cacheAlg == 'MFU' or l1cacheAlg == 'LRU2':
            metal1cache2 = np.zeros((l1setSize, l1ways), dtype=int)
        if l1cacheAlg == 'LRFU':
            metal1cache = np.zeros((l1setSize, l1ways), dtype=float)
        else:
            metal1cache = np.zeros((l1setSize, l1ways), dtype=int)
        setl1cache = np.zeros(l1setSize, dtype=int)
        wayl1cache = np.zeros((l1setSize, l1ways), dtype=int)

        # print(l1cache, l1cache[0][0][0] == 0)
        print(l1setSize)

        print(len(trace))
        # trace = trace[:100]
        # sets = [0] * len(trace)

        blockMask = int(''.join(['1'] * blockBits), 2)
        l1setMask = int(''.join(['1'] * l1setBits), 2)
        print(format(blockMask, "b"), format(l1setMask, "b"))
        
        # setup l2 cache
        l2cache = np.zeros((l2setSize, ways, blockSize), dtype=int)

        if l2cacheAlg == 'LFU' or l2cacheAlg == 'MFU' or l2cacheAlg == 'LRU2':
            metal2cache2 = np.zeros((l2setSize, ways), dtype=int)
        if l2cacheAlg == 'LRFU':
            metal2cache = np.zeros((l2setSize, ways), dtype=float)
        else:
            metal2cache = np.zeros((l2setSize, ways), dtype=int)
        setl2cache = np.zeros(l2setSize, dtype=int)
        wayl2cache = np.zeros((l2setSize, ways), dtype=int)

        # print(l2cache, l2cache[0][0][0] == 0)
        print(l2setSize)

        print(len(trace))
        # trace = trace[:100]
        # sets = [0] * len(trace)

        blockMask = int(''.join(['1'] * blockBits), 2)
        l2setMask = int(''.join(['1'] * l2setBits), 2)
        print(format(blockMask, "b"), format(l2setMask, "b"))

        hit = 0
        miss = 0

        for t in range(len(trace)):
            if t % 1000 == 0:
                print(t, len(trace), t / len(trace) * 100)
            addr = int(trace[t].split(' ')[1])
            # print('Address',format(a, "b"))
            block = addr & blockBits
            l1s = (addr >> l1setBits) & l1setMask
            l1tag = (addr >> l1setBits) >> blockBits

            setl1cache[l1s] += 1

            found = False
            for w in range(len(l1cache[l1s])):
                for b in l1cache[l1s][w]:
                    if l1tag == b:
                        hit += 1
                        found = True
                        if l1cacheAlg == 'LRU' or l1cacheAlg == 'MRU':
                            metal1cache[l1s][w] = t
                        elif l1cacheAlg == 'LRU2':
                            # store second most recent timestamp
                            metal1cache2[l1s][w] = metal1cache[l1s][w]
                            metal1cache[l1s][w] = t
                        elif l1cacheAlg == 'LFU' or l1cacheAlg == 'MFU':
                            metal1cache[l1s][w] += 1
                        elif l1cacheAlg == 'FIFO' or l1cacheAlg == 'LIFO':
                            metal1cache[l1s][w] = max(metal1cache[l1s]) + 1
                        elif l1cacheAlg == 'PLRU':
                            metal1cache[l1s][0] = (metal1cache[l1s][0] + 1) % l1ways
                        elif l1cacheAlg == 'LRFU':
                            for i in range(len(metal1cache[l1s])):
                                if l1cache[l1s][i][0] != 0 and i != w:
                                    metal1cache[l1s][i] = (2 ** (-1 * lamb)) * metal1cache[l1s][i]
                                else:
                                    metal1cache[l1s][i] = 1 + (2 ** (-1 * lamb)) * metal1cache[l1s][i]
                        break
            if not found:
                # go to l2 first
                block = addr & blockBits
                l2s = (addr >> l2setBits) & l2setMask
                l2tag = (addr >> l2setBits) >> blockBits
                for w in range(len(l2cache[l2s])):
                    for b in l2cache[l2s][w]:
                        if l2tag == b:
                            hit += 1
                            found = True
                            if l2cacheAlg == 'LRU' or l2cacheAlg == 'MRU':
                                metal2cache[l2s][w] = t
                            elif l2cacheAlg == 'LRU2':
                                # store second most recent timestamp
                                metal2cache2[l2s][w] = metal2cache[l2s][w]
                                metal2cache[l2s][w] = t
                            elif l2cacheAlg == 'LFU' or l2cacheAlg == 'MFU':
                                metal2cache[l2s][w] += 1
                            elif l2cacheAlg == 'FIFO' or l2cacheAlg == 'LIFO':
                                metal2cache[l2s][w] = max(metal2cache[l2s]) + 1
                            elif l2cacheAlg == 'PLRU':
                                metal2cache[l2s][0] = (metal2cache[l2s][0] + 1) % l2ways
                            elif l2cacheAlg == 'LRFU':
                                for i in range(len(metal2cache[l2s])):
                                    if l2cache[l2s][i][0] != 0 and i != w:
                                        metal2cache[l2s][i] = (2 ** (-1 * lamb)) * metal2cache[l2s][i]
                                    else:
                                        metal2cache[l2s][i] = 1 + (2 ** (-1 * lamb)) * metal2cache[l2s][i]
                            break
                if found:
                    # write to l1?
                    pass
                if not found:
                    # replace in l2
                    miss += 1
                    w = -1
                    for i in range(l1ways):
                        if l2cache[l2s][i][0] == 0:
                            w = i
                            break
                    if w == -1:
                        if l2cacheAlg == 'R':
                            w = random.randint(0, len(l2cache[l2s]) - 1)
                        elif l2cacheAlg == 'LRU' or l2cacheAlg == 'LIFO' or l2cacheAlg == 'LRFU':
                            w = np.argmin(metal2cache[l2s])
                        elif l2cacheAlg == 'LRU2':
                            w = np.argmin(metal2cache2[l2s])
                        elif l2cacheAlg == 'MRU' or l2cacheAlg == 'FIFO':
                            w = np.argmax(metal2cache[l2s])
                        elif l2cacheAlg == 'LFU':
                            # if there are multiple values decide by LRU
                            x = np.where(metal2cache[l2s] == metal2cache[l2s].min())
                            x = x[0]
                            w = x[0]
                            for y in range(len(x)):
                                if metal2cache2[s][x[y]] < metal2cache2[s][w]:
                                    w = x[y]
                        elif l2cacheAlg == 'MFU':
                            # if there are multiple values decide by LRU
                            x = np.where(metal2cache[s] == metal2cache[s].max())
                            x = x[0]
                            w = x[0]
                            for y in range(len(x)):
                                if metal2cache2[s][x[y]] < metal2cache2[s][w]:
                                    w = x[y]
                        elif l2cacheAlg == 'PLRU':
                            w = metal2cache[s][0]

                    if l2cacheAlg == 'LRU' or l2cacheAlg == 'MRU':
                        metal2cache[s][w] = t
                    elif l2cacheAlg == 'LRU2':
                        metal2cache2[s][w] = t
                        metal2cache[s][w] = t
                    elif l2cacheAlg == 'LFU' or l2cacheAlg == 'MFU':
                        metal2cache[s][w] = 1
                        metal2cache2[s][w] = t
                    elif l2cacheAlg == 'LIFO' or l2cacheAlg == 'FIFO':
                        metal2cache[s][w] = max(metal2cache[s]) + 1
                    elif l2cacheAlg == 'PLRU':
                        metal2cache[s][w] = 0
                    elif l2cacheAlg == 'LRFU':
                        for i in range(len(metal2cache[s])):
                            if i != w:
                                metal2cache[s][i] = (2 ** (-1 * lamb)) * metal2cache[s][i]
                            else:
                                metal2cache[s][i] = 0

                    wayl2cache[s][w] += 1
                    for b in range(len(l2cache[s][w])):
                        l2cache[s][w][b] = tag
                        
                    # replace in l1
                    miss += 1
                    w = -1
                    for i in range(l1ways):
                        if l1cache[s][i][0] == 0:
                            w = i
                            break
                    if w == -1:
                        if l1cacheAlg == 'R':
                            w = random.randint(0, len(l1cache[s]) - 1)
                        elif l1cacheAlg == 'LRU' or l1cacheAlg == 'LIFO' or l1cacheAlg == 'LRFU':
                            w = np.argmin(metal1cache[s])
                        elif l1cacheAlg == 'LRU2':
                            w = np.argmin(metal1cache2[s])
                        elif l1cacheAlg == 'MRU' or l1cacheAlg == 'FIFO':
                            w = np.argmax(metal1cache[s])
                        elif l1cacheAlg == 'LFU':
                            # if there are multiple values decide by LRU
                            x = np.where(metal1cache[s] == metal1cache[s].min())
                            x = x[0]
                            w = x[0]
                            for y in range(len(x)):
                                if metal1cache2[s][x[y]] < metal1cache2[s][w]:
                                    w = x[y]
                        elif l1cacheAlg == 'MFU':
                            # if there are multiple values decide by LRU
                            x = np.where(metal1cache[s] == metal1cache[s].max())
                            x = x[0]
                            w = x[0]
                            for y in range(len(x)):
                                if metal1cache2[s][x[y]] < metal1cache2[s][w]:
                                    w = x[y]
                        elif l1cacheAlg == 'PLRU':
                            w = metal1cache[s][0]
    
                    if l1cacheAlg == 'LRU' or l1cacheAlg == 'MRU':
                        metal1cache[s][w] = t
                    elif l1cacheAlg == 'LRU2':
                        metal1cache2[s][w] = t
                        metal1cache[s][w] = t
                    elif l1cacheAlg == 'LFU' or l1cacheAlg == 'MFU':
                        metal1cache[s][w] = 1
                        metal1cache2[s][w] = t
                    elif l1cacheAlg == 'LIFO' or l1cacheAlg == 'FIFO':
                        metal1cache[s][w] = max(metal1cache[s]) + 1
                    elif l1cacheAlg == 'PLRU':
                        metal1cache[s][w] = 0
                    elif l1cacheAlg == 'LRFU':
                        for i in range(len(metal1cache[s])):
                            if i != w:
                                metal1cache[s][i] = (2 ** (-1 * lamb)) * metal1cache[s][i]
                            else:
                                metal1cache[s][i] = 0
    
                    wayl1cache[s][w] += 1
                    for b in range(len(l1cache[s][w])):
                        l1cache[s][w][b] = tag


        hitTotal += hit
        missTotal += miss
        setl1cacheTotal = setl1cacheTotal + setl1cache
        wayl1cacheTotal = wayl1cacheTotal + wayl1cache

        print(metal1cache)

        print(hit)
        print(miss)
        print(hit / (hit + miss))
        info[file] = {}
        info[file]['hits'] = hit
        info[file]['miss'] = miss
        info[file]['setl1cache'] = setl1cache
        info[file]['wayl1cache'] = wayl1cache
        info[file]['metal1cache'] = metal1cache
        if l1cacheAlg == 'LFU' or l1cacheAlg == 'MFU' or l1cacheAlg == 'LRU2':
            info[file]['metal1cache2'] = metal1cache2

    info['hitTotal'] = hitTotal
    info['missTotal'] = missTotal
    info['setl1cacheTotal'] = setl1cacheTotal
    info['wayl1cacheTotal'] = wayl1cacheTotal

    with open('./Results/{}_{}KB_{}way.pickle'.format(sys.argv[2], sys.argv[3], sys.argv[4]), 'wb') as f:
        pickle.dump(info, f)


'''
plt.bar(range(setSize), setl1cache)
plt.show()

print(len(wayl1cache), len(wayl1cache[0]))
for i in range(setSize // 64):
    plt.subplot(1, setSize // 64, i + 1)
    plt.imshow(wayl1cache[i * 64:(i + 1) * 64], cmap='Reds', interpolation='nearest')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.show()
'''
