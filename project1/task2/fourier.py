import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####
def dft(img):
    ft = np.empty_like(img, dtype=np.complex128)
    for l in range(img.shape[0]):
        for k in range(img.shape[1]):
            sum = 0
            for j in range(img.shape[0]):
                for i in range(img.shape[1]):
                    sum += img[j, i] * np.exp(-(2 * 1j * np.pi * ((k * i / img.shape[1]) + (l * j / img.shape[0]))))
            ft[l, k] = sum
    return ft

def idft(img):
    ft = np.empty_like(img, dtype=np.complex128)
    for l in range(img.shape[0]):
        for k in range(img.shape[1]):
            sum = 0
            for j in range(img.shape[0]):
                for i in range(img.shape[1]):
                    sum += img[j, i] * np.exp(2 * 1j * np.pi * ((k * i / img.shape[1]) + (l * j / img.shape[0])))
            ft[l, k] = sum / (img.shape[0] * img.shape[1])
    return ft

def fshift(spectrum):
    spectrum = np.roll(spectrum, spectrum.shape[0] // 2, axis=0)
    spectrum = np.roll(spectrum, spectrum.shape[1] // 2, axis=1)
    return spectrum

def ifshift(spectrum):
    spectrum = np.roll(spectrum, -(spectrum.shape[0] // 2), axis=0)
    spectrum = np.roll(spectrum, -(spectrum.shape[1] // 2), axis=1)
    return spectrum

def get_distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)

def fm_spectrum(img):
    s = np.fft.fft2(img)
    #s1 = dft(img)
    s = np.log(np.abs(s))
    s = fshift(s)
    return s

def low_pass_filter(img, th=10):
    filter = np.fft.fft2(img)
    #s1 = dft(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if get_distance(filter.shape[0] // 2, filter.shape[1] // 2, i, j) > th:
                filter[j, i] = 1
    filter = ifshift(filter)
    #s2 = idft(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

def high_pass_filter(img, th=30):
    filter = np.fft.fft2(img)
    #s1 = dft(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if get_distance(filter.shape[0] // 2, filter.shape[1] // 2, i, j) < th:
                filter[j, i] = 1
    filter = ifshift(filter)
    #s2 = idft(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

def denoise1(img):
    inner_ring1 = 68
    outer_ring1 = inner_ring1 + 15

    inner_ring2 = 149
    outer_ring2 = inner_ring2 + 30

    inner_ring3 = 215
    outer_ring3 = inner_ring3 + 40

    inner_ring4 = 285
    outer_ring4 = inner_ring4 + 30

    inner_ring5 = 105
    outer_ring5 = inner_ring5 + 15
    filter = np.fft.fft2(img)
    #s1 = dft(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if ((inner_ring1 < get_distance(filter.shape[0] // 2, filter.shape[1] // 2, j, i) < outer_ring1 or
                    inner_ring2 < get_distance(filter.shape[0] // 2, filter.shape[1] // 2, j, i) < outer_ring2 or
                    inner_ring3 < get_distance(filter.shape[0] // 2, filter.shape[1] // 2, j, i) < outer_ring3 or
                    inner_ring4 < get_distance(filter.shape[0] // 2, filter.shape[1] // 2, j, i) < outer_ring4) and \
                    not(filter.shape[0] // 2 - 2 <= j <= filter.shape[1] // 2 + 2) and
                    not(filter.shape[1] // 2 - 2 <= i <= filter.shape[0] // 2 + 2)) or \
                    inner_ring5 < get_distance(filter.shape[0] // 2, filter.shape[1] // 2, j, i) < outer_ring5:
                filter[j, i] = 1
    filter = ifshift(filter)
    #s2 = idft(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

def denoise2(img):
    inner_ring = 40
    outer_ring = inner_ring + 4
    filter = np.fft.fft2(img)
    #s1 = dft(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if inner_ring < get_distance(filter.shape[0] // 2, filter.shape[1] // 2, j, i) < outer_ring:
                filter[j, i] = 1
    filter = ifshift(filter)
    #s2 = idft(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

#################
if __name__ == '__main__':
    img = cv2.imread('task2_sample.png', cv2.IMREAD_GRAYSCALE)
    cor1 = cv2.imread('task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)
    cor2 = cv2.imread('task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)

    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    # cv2.imwrite('denoised1.png', denoise1(cor1))
    # cv2.imwrite('denoised2.png', denoise2(cor2))
    lpf = low_pass_filter(img)
    hpf = high_pass_filter(img)
    d1 = denoise1(cor1)
    d2 = denoise2(cor2)

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), lpf, 'Low-pass')
    drawFigure((2,7,3), hpf, 'High-pass')
    drawFigure((2,7,4), cor1, 'Noised')
    drawFigure((2,7,5), d1, 'Denoised')
    drawFigure((2,7,6), cor2, 'Noised')
    drawFigure((2,7,7), d2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(lpf), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(hpf), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(cor1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(d1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(cor2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(d2), 'Spectrum')

    plt.show()
