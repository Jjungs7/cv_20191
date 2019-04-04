import cv2 as cv
import math
import numpy as np
from task1.utils import calculate_rms

images = [
    ['task1_sample1_clean.png', 'task1_sample1_noise.png'],
    ['task1_sample2_clean.png', 'task1_sample2_noise.png'],
    ['task1_sample3_clean.png', 'task1_sample3_noise.png'],
    ['task1_sample4_clean.png', 'task1_sample4_noise.png'],
    ['test1_clean.png', 'test1_noise.png'],
    ['test2_clean.png', 'test2_noise.png'],
    ['test3_clean.png', 'test3_noise.png'],
    ['test4_clean.png', 'test4_noise.png'],
    ['test5_clean.png', 'test5_noise.png']
]


def task1(src_img_path, clean_img_path, dst_img_path):
    results = []
    clean_img = cv.imread(clean_img_path, cv.IMREAD_COLOR)
    img = np.array(cv.imread(src_img_path, cv.IMREAD_COLOR))
    print()
    print(f'Computing image {src_img_path}')
    print(f'The original rms is = \t{calculate_rms(clean_img, img)}')
    print('-----------------------------------------------------------')

    def get_best_filter(kernel_size, sigma_s, sigma_r):
        print(f'with configs k: {kernel_size}, ss: {sigma_s}, sr: {sigma_r}')
        img_bilateral = apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r)
        img_median = apply_median_filter(img, kernel_size)
        img_average = apply_average_filter(img, kernel_size)
        img_mixed = apply_median_filter(img_bilateral, kernel_size)

        rms_bilateral = calculate_rms(clean_img, img_bilateral)
        rms_median = calculate_rms(clean_img, img_median)
        rms_average = calculate_rms(clean_img, img_average)
        rms_mixed = calculate_rms(clean_img, img_mixed)

        final_img = img_bilateral if rms_bilateral < rms_average else (img_average if rms_average < rms_median else (img_median if rms_median < rms_mixed else img_mixed))
        final_rms = calculate_rms(clean_img, final_img)
        results.append(final_rms)
        if min(results) == final_rms:
            cv.imwrite(dst_img_path, final_img)
        print(f'best rms is =\t\t\t{final_rms}')

    configs = [
        (3, 75, 75),
        (3, 90, 90),
        (3, 95, 95),
        (5, 75, 75),
        (5, 90, 90),
        (9, 75, 75),
        (9, 90, 90),
        (15, 75, 75),
        (15, 90, 90)
    ]

    for kernel_size, sigma_s, sigma_r in configs:
        get_best_filter(kernel_size, sigma_s, sigma_r)
    print('-----------------------------------------------------------')


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
    def l1_distance(x1, y1, z1, x2, y2, z2):
        return np.abs(x2-x1) + np.abs(y2-y1) + np.abs(z2-z1)

    def gaussian(x, sigma):
        return math.exp(-(x ** 2 / (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)

    def bilateral(src, y, x, diam, sigma_s, sigma_r):
        b, g, r = src
        pixel_b = 0
        pixel_g = 0
        pixel_r = 0
        rad = diam // 2
        wp = 0
        for j in range(-rad, rad+1):
            for i in range(-rad, rad+1):
                roi_y = y+j
                roi_x = x+i
                if roi_y < 0 or b.shape[0] <= roi_y or roi_x < 0 or b.shape[1] <= roi_x:
                    continue
                g_s = gaussian(np.sqrt((roi_x - x) ** 2 + (roi_y - y) ** 2), sigma_s)
                g_r = gaussian(l1_distance(np.int(b[roi_y, roi_x]), np.int(g[roi_y, roi_x]), np.int(r[roi_y, roi_x]),
                                           np.int(b[y, x]), np.int(g[y, x]), np.int(r[y, x])), sigma_r)
                w = g_s * g_r
                pixel_b += w * b[roi_y, roi_x]
                pixel_g += w * g[roi_y, roi_x]
                pixel_r += w * r[roi_y, roi_x]
                wp += w

        return [pixel_b // wp, pixel_g // wp, pixel_r // wp]

    b, g, r = [c for c in cv.split(img)]
    new_colors = np.full_like(img, 127)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            res = bilateral((b, g, r), y, x, kernel_size, sigma_s, sigma_r)
            new_colors[y, x, 0] = res[0]
            new_colors[y, x, 1] = res[1]
            new_colors[y, x, 2] = res[2]

    return cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))


for clean, noise in [i for i in images]:
    task1(f'./data/{noise}', f'./data/{clean}', f'./res/{noise}')

img_num = 1
img_name = 'task1_sample' + str(img_num)
#task1(f'./data/{img_name}_noise.png', f'./data/{img_name}_clean.png', f'./res/{img_name}_noise.png')
