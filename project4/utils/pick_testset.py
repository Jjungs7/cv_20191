import os
import random

def pick_set(percentage, TRAIN_DIRS, TEST_DIRS):
    base = 0
    for idx, d in enumerate(TRAIN_DIRS):
        for f in os.listdir(d):
            rand = random.randrange(base, base+101)
            if base <= rand < base + percentage:
                os.rename(os.path.join(TRAIN_DIRS[idx], f), os.path.join(TEST_DIRS[idx], f))

