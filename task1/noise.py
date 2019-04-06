import cv2 as cv
import math
import numpy as np
import os
from multiprocessing import Process
from task1.utils import calculate_rms

processes = []
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
    clean_img = cv.imread(clean_img_path, cv.IMREAD_COLOR)
    img = np.array(cv.imread(src_img_path, cv.IMREAD_COLOR))
    paths = os.path.split(src_img_path)
    path = paths[len(paths) - 1]
    filename, extension = os.path.splitext(path)
    path = f'{filename}.log'
    f = open(f'./log/{path}', 'w')
    f.write(f'Computing image {src_img_path}\n')
    f.write(f'rms at first =\t{calculate_rms(clean_img, img)}\n')
    f.write('-----------------------------------------------------------\n')

    best_img = apply_average_filter(img, 3)
    filter_type = 'average'
    for kernel_size in [3, 5, 7, 9, 11, 13, 15]:
        filters = [
            (apply_average_filter(img, kernel_size), 'average'),
            (apply_median_filter(img, kernel_size), 'median'),
            (apply_custom_filter(img, kernel_size), 'custom')
        ]

        for filtered_img, ft in filters:
            if calculate_rms(filtered_img, clean_img) < calculate_rms(best_img, clean_img):
                best_img = filtered_img
                filter_type = ft


    def get_best_filter(best_img, filter_type, kernel_size, sigma_s, sigma_r):
        img_bilateral = apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r)
        filters = [
            (img_bilateral, 'bilateral'),
            (apply_median_filter(img_bilateral, kernel_size), 'mixed')
        ]

        for filtered_img, ft in filters:
            if calculate_rms(filtered_img, clean_img) < calculate_rms(best_img, clean_img):
                best_img = filtered_img
                filter_type = ft

        f.write(f'best rms is =\t{calculate_rms(clean_img, best_img)}\n'
              f'with configs k: {kernel_size}, ss: {sigma_s}, sr: {sigma_r}. Filter type=\t{filter_type}\n')

        return best_img

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
        best_img = get_best_filter(best_img, filter_type, kernel_size, sigma_s, sigma_r)

    cv.imwrite(dst_img_path, best_img)
    f.write('-----------------------------------------------------------\n')
    f.close()


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
                kernel = color[y:y+kernel_size, x:x+kernel_size]
                res = round(np.sum(kernel, dtype=float) / (kernel_size ** 2))
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
                kernel = color[y:y + kernel_size, x:x + kernel_size]
                res = np.median(kernel)
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
                dy = y+j
                dx = x+i
                if dy < 0 or b.shape[0] <= dy or dx < 0 or b.shape[1] <= dx:
                    continue
                g_s = gaussian(np.sqrt((dx - x) ** 2 + (dy - y) ** 2), sigma_s)
                g_r = gaussian(l1_distance(np.int(b[dy, dx]), np.int(g[dy, dx]), np.int(r[dy, dx]),
                                           np.int(b[y, x]), np.int(g[y, x]), np.int(r[y, x])), sigma_r)
                w = g_s * g_r
                pixel_b += w * b[dy, dx]
                pixel_g += w * g[dy, dx]
                pixel_r += w * r[dy, dx]
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


def apply_custom_filter(img, kernel_size):
    histogram_set_size = 48
    allow_limit = 3
    def get_img_histogram(k):
        sections = 256 // histogram_set_size
        if 256 % histogram_set_size > 0:
            sections += 1
        histogram = [0]*sections
        for j in range(k.shape[0]):
            for i in range(k.shape[1]):
                histogram[k[j, i] // histogram_set_size] += 1
        return histogram

    def is_relevant(h, pixel):
        return h[pixel // histogram_set_size] > allow_limit

    def convolution(k, y, x):
        filter = np.ones_like(k)
        filter[y, x] = 0
        res = np.sum((k * filter)) // (filter.shape[0] * filter.shape[1] - 1)
        return res

    new_colors = np.full_like(img, 0)
    for idx, color in enumerate([c for c in cv.split(img)]):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                radius = kernel_size // 2
                center_x, center_y = radius, radius
                x1, y1, x2, y2 = x - radius, y - radius,  x + radius, y + radius
                if x1 < 0:
                    x1 = 0
                    center_x = x1
                if x2 >= img.shape[1]:
                    x2 = img.shape[1] - 1
                    center_x = x2 - x1
                if y1 < 0:
                    y1 = 0
                    center_y = y1
                if y2 >= img.shape[0]:
                    y2 = img.shape[0] - 1
                    center_y = y2 - y1

                kernel = color[y1:y2+1, x1:x2+1]
                histogram = get_img_histogram(kernel)
                if not is_relevant(histogram, kernel[center_y, center_x]):
                    res = convolution(kernel, center_y, center_x)
                    new_colors[y, x, idx] = res
                else:
                    new_colors[y, x, idx] = color[y, x]

    return cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))


for clean, noise in [i for i in images]:
    processes.append(Process(target=task1, args=(f'./data/{noise}', f'./data/{clean}', f'./res/{noise}')))

for process in processes:
    process.start()

for process in processes:
    process.join()

img_num = 1
img_name = 'task1_sample' + str(img_num)
#task1(f'./data/{img_name}_noise.png', f'./data/{img_name}_clean.png', f'./res/{img_name}_noise.png')

"""
res = []
for kernel_size in range(3, 16):
    if kernel_size % 2 == 0:
        continue

    filename = 'test5'
    custom_img = apply_custom_filter(cv.imread(f'./data/{filename}_noise.png', cv.IMREAD_COLOR), kernel_size)
    rms_custom = calculate_rms(custom_img, cv.imread(f'./data/{filename}_clean.png'))
    print(rms_custom)

    res.append(rms_custom)
    if min(res) == rms_custom:
        cv.imwrite(f'./res/{filename}_noise.png', custom_img)
"""
