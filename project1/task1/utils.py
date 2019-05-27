import numpy as np

"""
Calculates RMS error between two images. Two images should have same sizes.
"""
def calculate_rms(img1, img2):
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have sime sizes.")

    diff = np.abs(img1 - img2)
    return np.sqrt(np.mean(diff ** 2))


"""
Calculates RMS error between two cropped images. Two images should have same sizes.
"""
def calculate_rms_cropped(img1, img2):
    H, W, C = img1.shape
    cut_size = 20

    img1 = img1[cut_size:H - cut_size, cut_size:W - cut_size]
    img2 = img2[cut_size:H - cut_size, cut_size:W - cut_size]

    diff = np.abs(img1 - img2)
    return np.sqrt(np.mean(diff ** 2))