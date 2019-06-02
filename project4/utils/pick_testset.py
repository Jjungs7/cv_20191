import os
import random

def pick_set(percentage, TRAIN_DIRS, TEST_DIRS):
    for idx, d in enumerate(TRAIN_DIRS):
        for f in os.listdir(d):
            rand = random.randrange(1, 101)
            if 33 <= rand < 33 + percentage:
                os.rename(os.path.join(TRAIN_DIRS[idx], f), os.path.join(TEST_DIRS[idx], f))

