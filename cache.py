import numpy as np
from math import log2
import random


class Cache:
    def __init__(self, cacheAlg, cSize, ways, bSize = 64//8, lamb=0.5):
        self.cacheAlg = cacheAlg
        self.cacheSize = cSize
        self.ways = ways

        self.lamb = lamb  # 0=>LFU 1=>LRU

        self.blockSize = bSize  # 64 bits, 8 bits per byte

        self.addressSize = 48 // 8  # bytes

        self.setSize = cSize // bSize // ways

        self.blockBits = int(log2(bSize))
        self.setBits = int(log2(self.setSize))

        self.cache = np.zeros((self.setSize, ways, bSize), dtype=int)

        if cacheAlg == 'LFU' or cacheAlg == 'MFU' or cacheAlg == 'LRU2':
            self.metaCache2 = np.zeros((self.setSize, ways), dtype=int)
        if cacheAlg == 'LRFU':
            self.metaCache = np.zeros((self.setSize, ways), dtype=float)
        else:
            self.metaCache = np.zeros((self.setSize, ways), dtype=int)
        self.setCache = np.zeros(self.setSize, dtype=int)
        self.wayCache = np.zeros((self.setSize, ways), dtype=int)

        self.blockMask = int(''.join(['1'] * self.blockBits), 2)
        self.setMask = int(''.join(['1'] * self.setBits), 2)
        
        self.hit = 0
        self.miss = 0

    def reset(self):
        self.cache = np.zeros((self.setSize, self.ways, self.blockSize), dtype=int)

        if self.cacheAlg == 'LFU' or self.cacheAlg == 'MFU' or self.cacheAlg == 'LRU2':
            self.metaCache2 = np.zeros((self.setSize, self.ways), dtype=int)
        if self.cacheAlg == 'LRFU':
            self.metaCache = np.zeros((self.setSize, self.ways), dtype=float)
        else:
            self.metaCache = np.zeros((self.setSize, self.ways), dtype=int)
        self.setCache = np.zeros(self.setSize, dtype=int)
        self.wayCache = np.zeros((self.setSize, self.ways), dtype=int)

        self.hit = 0
        self.miss = 0
        
    '''
    Search through cache for address
    return True if found
    otherwise False
    '''
    def find(self, address):
        block = address & self.blockBits
        setNum= (address >> self.setBits) & self.setMask
        tag = (address >> self.setBits) >> self.blockBits

        self.setCache[setNum] += 1

        found = False
        for w in range(len(self.cache[setNum])):
            for b in self.cache[setNum][w]:
                if tag == b:
                    self.hit += 1
                    found = True
                    if self.cacheAlg == 'LRU' or self.cacheAlg == 'MRU':
                        self.metaCache[setNum][w] = self.hit + self.miss
                    elif self.cacheAlg == 'LRU2':
                        # store second most recent timestamp
                        self.metaCache2[setNum][w] = self.metaCache[setNum][w]
                        self.metaCache[setNum][w] = self.hit + self.miss
                    elif self.cacheAlg == 'LFU' or self.cacheAlg == 'MFU':
                        self.metaCache[setNum][w] += 1
                    elif self.cacheAlg == 'FIFO' or self.cacheAlg == 'LIFO':
                        self.metaCache[setNum][w] = max(self.metaCache[setNum]) + 1
                    elif self.cacheAlg == 'PLRU':
                        self.metaCache[setNum][0] = (self.metaCache[setNum][0] + 1) % self.ways
                    elif self.cacheAlg == 'LRFU':
                        for i in range(len(self.metaCache[setNum])):
                            if self.cache[setNum][i][0] != 0 and i != w:
                                self.metaCache[setNum][i] = (2 ** (-1 * self.lamb)) * self.metaCache[setNum][i]
                            else:
                                self.metaCache[setNum][i] = 1 + (2 ** (-1 * self.lamb)) * self.metaCache[setNum][i]
                    break
        return found
    
    '''
    Load data into the cache based off cache alg
    '''
    def load(self, address):
        block = address & self.blockBits
        setNum= (address >> self.setBits) & self.setMask
        tag = (address >> self.setBits) >> self.blockBits
        self.miss += 1
        w = -1
        for i in range(self.ways):
            if self.cache[setNum][i][0] == 0:
                w = i
                break
        if w == -1:
            if self.cacheAlg == 'R':
                w = random.randint(0, len(self.cache[setNum]) - 1)
            elif self.cacheAlg == 'LRU' or self.cacheAlg == 'LIFO' or self.cacheAlg == 'LRFU':
                w = np.argmin(self.metaCache[setNum])
            elif self.cacheAlg == 'LRU2':
                w = np.argmin(self.metaCache2[setNum])
            elif self.cacheAlg == 'MRU' or self.cacheAlg == 'FIFO':
                w = np.argmax(self.metaCache[setNum])
            elif self.cacheAlg == 'LFU':
                # if there are multiple values decide by LRU
                x = np.where(self.metaCache[setNum] == self.metaCache[setNum].min())
                x = x[0]
                w = x[0]
                for y in range(len(x)):
                    if self.metaCache2[setNum][x[y]] < self.metaCache2[setNum][w]:
                        w = x[y]
            elif self.cacheAlg == 'MFU':
                # if there are multiple values decide by LRU
                x = np.where(self.metaCache[setNum] == self.metaCache[setNum].max())
                x = x[0]
                w = x[0]
                for y in range(len(x)):
                    if self.metaCache2[setNum][x[y]] < self.metaCache2[setNum][w]:
                        w = x[y]
            elif self.cacheAlg == 'PLRU':
                w = self.metaCache[setNum][0]

        if self.cacheAlg == 'LRU' or self.cacheAlg == 'MRU':
            self.metaCache[setNum][w] = self.hit + self.miss
        elif self.cacheAlg == 'LRU2':
            self.metaCache2[setNum][w] = self.hit + self.miss
            self.metaCache[setNum][w] = self.hit + self.miss
        elif self.cacheAlg == 'LFU' or self.cacheAlg == 'MFU':
            self.metaCache[setNum][w] = 1
            self.metaCache2[setNum][w] = self.hit + self.miss
        elif self.cacheAlg == 'LIFO' or self.cacheAlg == 'FIFO':
            self.metaCache[setNum][w] = max(self.metaCache[setNum]) + 1
        elif self.cacheAlg == 'PLRU':
            self.metaCache[setNum][w] = 0
        elif self.cacheAlg == 'LRFU':
            for i in range(len(self.metaCache[setNum])):
                if i != w:
                    self.metaCache[setNum][i] = (2 ** (-1 * self.lamb)) * self.metaCache[setNum][i]
                else:
                    self.metaCache[setNum][i] = 0

        self.wayCache[setNum][w] += 1
        for b in range(len(self.cache[setNum][w])):
            self.cache[setNum][w][b] = tag