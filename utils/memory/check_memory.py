#! /usr/bin/env python3
import sys

#CUDA Malloc,  pointer address: 0x2b2597bf0200, size: 32768 
#CUDA Free, pointer address: 0x2ac28dbe8400 

filename = sys.argv[1]

allocated = {}
watermark = -1000
biggest = -1000
actual = 0
total = 0

with open(filename, "r") as f:
    for line in f:
        if(line.startswith("CUDA Malloc")):
            tokens = line.replace(",", " ").split()
            adr = tokens[4]
            size = int(tokens[6])
            #print("%s, %s" % (adr, size))
            if(adr in allocated.keys()):
                raise Exception("Address already allocated")
            allocated[adr] = size
            actual += size
            watermark = max(watermark, actual)
            biggest = max(biggest, size)
            total += size

        if(line.startswith("CUDA Free")):
            tokens = line.replace(",", " ").split()
            adr = tokens[4]
            #print("%s, %s" % (adr))
            if(adr not in allocated.keys()):
                raise Exception("Address not allocated cannot be freed")
            size = allocated[adr]
            actual -= size
            allocated.pop(adr)


print("Watermark %.2f MB" % (watermark / 2**20))
print("Total     %.2f MB" % (total / 2**20))
print("Biggest   %.2f MB" % (biggest / 2**20))
print("Nonfreed  %.2f MB" % (actual / 2**20))
print(allocated)




