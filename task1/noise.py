import cv2 as cv
import numpy as np
from utils import calculate_rms


def task1(src_img_path, clean_img_path, dst_img_path):
    EXEC_MODE = 'submission'
    clean_img = cv.imread(clean_img_path, cv.IMREAD_COLOR)
    img = np.array(cv.imread(src_img_path, cv.IMREAD_COLOR))
    if EXEC_MODE == 'development':
        import os
        paths = os.path.split(src_img_path)
        path = paths[len(paths) - 1]
        filename, extension = os.path.splitext(path)
        path = f'{filename}.log'
        f = open(f'./log/{path}', 'w')
        f.write(f'Computing image {src_img_path}\n')
        f.write(f'rms at first =\t{calculate_rms(clean_img, img)}\n')
        f.write('-----------------------------------------------------------\n')

    best_img = apply_average_filter(img, 3)
    best_filter_type = 'average'
    filter_type2 = 'average'
    filter_size = 3
    b52 = apply_bilateral_filter(img, 5, 90, 90)
    b52 = apply_bilateral_filter(b52, 5, 90, 90)
    for kernel_size in [3, 5, 7, 9, 11]:
        filters = [
            (apply_average_filter(img, kernel_size), 'average'),
            (apply_median_filter(img, kernel_size), 'median'),
            (apply_bilateral_filter(img, kernel_size, 90, 90), 'bilateral'),
            (apply_bilateral2_filter(img, kernel_size, 20, 30), 'bilateral2'),
            (apply_activate_filter(img, kernel_size), 'custom'),
            (apply_bilateral_filter(b52, kernel_size, 90, 90), 'mixed')
        ]

        for filtered_img, ft in filters:
            if calculate_rms(filtered_img, clean_img) < calculate_rms(best_img, clean_img):
                best_img = filtered_img
                best_filter_type = ft
                filter_type2 = ft
                filter_size = kernel_size

        if EXEC_MODE == 'development':
            f.write(f'best rms is =\t{calculate_rms(clean_img, best_img)}\tFilter type=\t{best_filter_type}\n\n')

    cv.imwrite(dst_img_path, filters[0][0])
    if EXEC_MODE == 'development':
        f.write(f'Compared with filter type:\t{filter_type2}, size:\t{filter_size}\n')
        f.write('-----------------------------------------------------------\n')
        f.close()


def apply_average_filter(img, kernel_size):
    padding = kernel_size // 2
    new_colors = np.full_like(img, 0)
    for idx, color in enumerate(np.array([c for c in cv.split(img)])):
        for _ in range(padding):
            color = np.c_[np.zeros(color.shape[0]), color]
            color = np.r_[[np.zeros(color.shape[1])], color]
            color = np.c_[color, np.zeros(color.shape[0])]
            color = np.r_[color, [np.zeros(color.shape[1])]]

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                kernel = color[y:y+kernel_size, x:x+kernel_size]
                res = round(np.sum(kernel, dtype=float) / (kernel_size ** 2))
                new_colors[y][x][idx] = res

    return cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))


def apply_median_filter(img, kernel_size):
    padding = kernel_size // 2
    new_colors = np.full_like(img, 0)
    for idx, color in enumerate(np.array([c for c in cv.split(img)])):
        for _ in range(padding):
            color = np.c_[np.zeros(color.shape[0]), color]
            color = np.r_[[np.zeros(color.shape[1])], color]
            color = np.c_[color, np.zeros(color.shape[0])]
            color = np.r_[color, [np.zeros(color.shape[1])]]

        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                kernel = color[y:y + kernel_size, x:x + kernel_size]
                res = np.median(kernel)
                new_colors[y][x][idx] = res

    return cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    def l1_distance(x1, y1, z1, x2, y2, z2):
        return np.abs(x2-x1) + np.abs(y2-y1) + np.abs(z2-z1)

    def gaussian(x, sigma):
        #return np.exp(-(x ** 2 / (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
        return np.exp(-((x / sigma) ** 2 / 2))

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
                # g_s 에는 euclidean distance가 들어가고
                # g_r 에는 b, g, r 을 각각 한개의 차원으로 잡아서 l1_distance를 구하고 넘긴다.
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


def apply_bilateral2_filter(img, kernel_size, sigma_s, sigma_r):
    def gaussian(x, sigma):
        #return np.exp(-(x ** 2 / (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
        return np.exp(-((x / sigma) ** 2 / 2))

    def bilateral(src, y, x, diam, sigma_s, sigma_r):
        pixel = 0
        rad = diam // 2
        wp = 0
        for j in range(-rad, rad+1):
            for i in range(-rad, rad+1):
                dy = y+j
                dx = x+i
                if dy < 0 or src.shape[0] <= dy or dx < 0 or src.shape[1] <= dx:
                    continue
                # g_s 에는 euclidean distance가 들어가고
                # g_r 에는 kernel 상에서 중심 점에 있는 픽셀값, 다른 점에 있는 픽셀값의 차이를 넘긴다.
                g_s = gaussian(np.sqrt((dx - x) ** 2 + (dy - y) ** 2), sigma_s)
                g_r = gaussian(np.abs(np.int(src[dy, dx]) - np.int(src[y, x])), sigma_r)
                w = g_s * g_r
                pixel += w * src[dy, dx]
                wp += w

        return pixel // wp

    colors = [c for c in cv.split(img)]
    new_colors = np.full_like(img, 127)
    for idx, color in enumerate(colors):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                res = bilateral(color, y, x, kernel_size, sigma_s, sigma_r)
                new_colors[y, x, idx] = res

    return cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))


def apply_activate_filter(img, kernel_size):
    def is_relevant(pixel):
        return 1 <= pixel <= 254

    def convolution(k):
        filter = np.ones_like(k)
        relevants = filter.shape[0] * filter.shape[1]
        for j in range(filter.shape[0]):
            for i in range(filter.shape[1]):
                if not is_relevant(k[j, i]):
                    filter[j, i] = 0
                    relevants -= 1
        if relevants > 0:
            new_pixel = np.sum(k * filter)
            new_pixel = new_pixel // relevants
            return new_pixel
        else:
            return 0

    new_colors = np.full_like(img, 0)
    for idx, color in enumerate([c for c in cv.split(img)]):
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                radius = kernel_size // 2
                center_x, center_y = radius, radius
                x1, y1, x2, y2 = x - radius, y - radius,  x + radius, y + radius
                # border에서 벗어날 경우 x1, x2, y1, y2 값을 조정하고
                # 중심점을 조정한다.
                if x1 < 0:
                    x1 = 0
                    center_x = x
                if x2 >= img.shape[1]:
                    x2 = img.shape[1] - 1
                if y1 < 0:
                    y1 = 0
                    center_y = y
                if y2 >= img.shape[0]:
                    y2 = img.shape[0] - 1

                # 구한 x1, x2, y1, y2로 커널을 잡는다
                kernel = color[y1:y2+1, x1:x2+1]
                # histogram으로 kernel의 중심점 픽셀값이 이미지와 관련이 있는지 확인한다
                if not is_relevant(kernel[center_y, center_x]):
                    # 중심점이 관련이 없을 경우 중심점을 배제한 average_filter을 적용한다
                    res = convolution(kernel)
                    new_colors[y, x, idx] = res
                else:
                    # 관련이 있는 픽셀이면 그대로 저장한다
                    new_colors[y, x, idx] = color[y, x]

    return cv.merge((new_colors[:, :, 0], new_colors[:, :, 1], new_colors[:, :, 2]))


"""
if __name__ == '__main__':
    c, n = 'test1_clean.png', 'test1_noise.png'
    clean_img = cv.imread(f'./data/{c}', cv.IMREAD_COLOR)
    src_img = cv.imread(f'./data/{n}', cv.IMREAD_COLOR)
    dst_path = f'./res/{n}'

    m_images = [
        (cv.imread('b_k5_2.png', cv.IMREAD_COLOR), 'b52_')
    ]

    def do_b(image, name, k, ss, sr):
        b = apply_bilateral_filter(image, k, ss, sr)
        print(f'{name}b: {calculate_rms(b, clean_img)}')
        cv.imwrite(f'{name}b.png', b)
    
    imgs = [
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

    processes = []
    images = [
        ['test1_clean.png', 'test1_noise.png'],
        ['test2_clean.png', 'test2_noise.png']
        #['test3_clean.png', 'test3_noise.png']
    ]

    
    from multiprocessing import Process
    for clean, noise in [i for i in images]:
        processes.append(Process(target=task1, args=(f'./data/{noise}', f'./data/{clean}', f'./res/{noise}')))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for clean, noise in images:
        c, n = cv.imread(f'./data/{clean}', cv.IMREAD_COLOR), cv.imread(f'./res/{noise}', cv.IMREAD_COLOR)
        print(calculate_rms(c, n))

    img4_noise = cv.imread('./data/test4_noise.png', cv.IMREAD_COLOR)
    img4_clean = cv.imread('./data/test4_clean.png', cv.IMREAD_COLOR)

    final_img2 = apply_activate_filter(img4_noise, 11)
    print(calculate_rms(final_img2, img4_clean))
    cv.imwrite('./res/test4_noise.png', final_img2)


    img5_noise = cv.imread('./data/test5_noise.png', cv.IMREAD_COLOR)
    img5_clean = cv.imread('./data/test5_clean.png', cv.IMREAD_COLOR)
    
    final_img = apply_average_filter(img5_noise, 5)
    print(calculate_rms(final_img, img5_clean))
    cv.imwrite('./res/test5_noise.png', final_img)
"""
