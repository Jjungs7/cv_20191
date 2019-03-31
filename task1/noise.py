import cv2 as cv
import math
import numpy as np
import threading
from task1.utils import calculate_rms

images = [
    ['task1_sample1_clean.png', 'task1_sample1_noise.png'],
    ['task1_sample2_clean.png', 'task1_sample2_noise.png'],
    ['task1_sample3_clean.png', 'task1_sample3_noise.png'],
    ['task1_sample4_clean.png', 'task1_sample4_noise.png']
]

kernel_size = 3
sigma_s = 20
sigma_r = 30

def task1(src_img_path, dst_img_path):
    original_img = cv.imread(src_img_path.replace('noise', 'clean'))
    img = np.array(cv.imread(src_img_path, cv.IMREAD_COLOR))
    print(f'The original rms is = {calculate_rms(original_img, img)}')

    new_img = apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r)
    #new_img = apply_median_filter(img, kernel_size)
    #new_img = apply_average_filter(img, kernel_size)

    cv.imwrite(dst_img_path, new_img)
    print(f'Final rms is = {calculate_rms(original_img, new_img)}')


"""
You should implement average filter convolution algorithm in this function.
It takes 2 arguments,
'img' is source image, and you should perform convolution with average filter.
'kernel_size' is a int value, which determines kernel size of average filter.

You should return result image.
"""
def apply_average_filter(img, kernel_size):
    lu_padding = math.floor((kernel_size + 1) / 2)
    rd_padding = math.ceil((kernel_size + 1) / 2)

    if img.shape[2] == 1:
        colors = img
    else:
        colors = np.array([c for c in cv.split(img)])

    new_colors = np.full_like(img, 0)
    for idx, color in enumerate(colors):
        for _ in range(lu_padding-1):
            color = np.c_[np.zeros(color.shape[0]), color]
            color = np.r_[[np.zeros(color.shape[1])], color]
        for _ in range(rd_padding-1):
            color = np.c_[color, np.zeros(color.shape[0])]
            color = np.r_[color, [np.zeros(color.shape[1])]]

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                roi = color[y:y+kernel_size, x:x+kernel_size]
                res = round(np.sum(roi, dtype=float) / (kernel_size ** 2))
                new_colors[y][x][idx] = res

    if img.shape[2] == 3:
        colors = cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))
    elif img.shape[2] == 1:
        colors = new_colors[0]
    return colors

"""
You should implement median filter convolution algorithm in this function.
It takes 2 arguments,
'img' is source image, and you should perform convolution with median filter.
'kernel_size' is a int value, which determines kernel size of median filter.

You should return result image.
"""
def apply_median_filter(img, kernel_size):
    lu_padding = math.floor((kernel_size + 1) / 2)
    rd_padding = math.ceil((kernel_size + 1) / 2)

    if img.shape[2] == 1:
        colors = img
    else:
        colors = np.array([c for c in cv.split(img)])

    new_colors = np.full_like(img, 0)
    for idx, color in enumerate(colors):
        for _ in range(lu_padding - 1):
            color = np.c_[np.zeros(color.shape[0]), color]
            color = np.r_[[np.zeros(color.shape[1])], color]
        for _ in range(rd_padding - 1):
            color = np.c_[color, np.zeros(color.shape[0])]
            color = np.r_[color, [np.zeros(color.shape[1])]]

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                roi = color[y:y + kernel_size, x:x + kernel_size]
                res = np.median(roi)
                new_colors[y][x][idx] = res

    if img.shape[2] == 3:
        colors = cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))
    elif img.shape[2] == 1:
        colors = new_colors[0]
    return colors


"""
You should implement convolution with additional filter.
You can use any filters for this function, except average, median filter.
It takes at least 2 arguments,
'img' is source image, and you should perform convolution with median filter.
'kernel_size' is a int value, which determines kernel size of average filter.
'sigma_s' is a int value, which is a sigma value for G_s
'sigma_r' is a int value, which is a sigma value for G_r

You can add more arguments for this function if you need.

You should return result image.
"""
def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):

    def gaussian(x, sigma):
        return math.exp(-(x ** 2 / (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)

    def bilateral(src, y, x, diam, sigma_s, sigma_r):
        new_i = 0
        r = diam // 2
        wp = 0
        for j in range(-r, r+1):
            for i in range(-r, r+1):
                roi_y = y+j
                roi_x = x+i
                if roi_y < 0 or src.shape[0] <= roi_y or roi_x < 0 or src.shape[1] <= roi_x:
                    continue
                g_s = gaussian(np.sqrt((roi_x - x) ** 2 + (roi_y - y) ** 2), sigma_s)
                g_r = gaussian(np.int(src[roi_y, roi_x]) - np.int(src[y, x]), sigma_r)
                w = g_s * g_r
                new_i += w * src[roi_y, roi_x]
                wp += w
        new_i /= wp
        return new_i

    if img.shape[2] == 1:
        colors = img
    else:
        colors = np.array([c for c in cv.split(img)])

    new_colors = np.full_like(img, 127)
    for idx, color in enumerate(colors):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                res = bilateral(color, y, x, kernel_size, sigma_s, sigma_r)
                new_colors[y, x, idx] = res

    if img.shape[2] == 3:
        colors = cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))
    elif img.shape[2] == 1:
        colors = new_colors[0]
    return colors


#for image in [noise[1] for noise in images]:
    #task1(f'./data/{image}', f'./res/{image}')
task1('./data/task1_sample2_noise.png', './res/task1_sample2_noise.png')
#cv.imshow('./res/task1_sample1_noise1.png', mat=cv.imread('./res/task1_sample1_noise1.png', cv.IMREAD_COLOR))
#cv.waitKey(0)
#cv.destroyAllWindows()

arr = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=int)
#print(arr[-3:, -3:])

#print(np.r_[arr, [np.zeros(arr.shape[1])]])
