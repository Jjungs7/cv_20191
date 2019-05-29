import os
import random

TRAIN_FAKE_DIR = '../data/train/fake'
TRAIN_GAN_DIR = '../data/train/gan'
TRAIN_REAL_DIR = '../data/train/real'

TEST_FAKE_DIR = '../data/test/fake'
TEST_GAN_DIR = '../data/test/gan'
TEST_REAL_DIR = '../data/test/real'

for i, dir in enumerate(os.listdir(TRAIN_FAKE_DIR)):
    rand = random.randrange(1, 101)
    if 50 <= rand <= 54 or 35 <= rand <= 39:
        os.rename(os.path.join(TRAIN_FAKE_DIR, dir), os.path.join(TEST_FAKE_DIR, dir))

for i, dir in enumerate(os.listdir(TRAIN_GAN_DIR)):
    rand = random.randrange(1, 101)
    if 50 <= rand <= 54 or 35 <= rand <= 39:
        os.rename(os.path.join(TRAIN_GAN_DIR, dir), os.path.join(TEST_GAN_DIR, dir))

for i, dir in enumerate(os.listdir(TRAIN_REAL_DIR)):
    rand = random.randrange(1, 101)
    if 50 <= rand <= 54 or 35 <= rand <= 39:
        os.rename(os.path.join(TRAIN_REAL_DIR, dir), os.path.join(TEST_REAL_DIR, dir))
