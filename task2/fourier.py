import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####
def dft(x):
    # TODO
    return x

def fshift(spectrum):
    spectrum = np.roll(spectrum, spectrum.shape[0] // 2, axis=0)
    spectrum = np.roll(spectrum, spectrum.shape[1] // 2, axis=1)
    return spectrum

def fm_spectrum(img):
    s = np.fft.fft2(img)
    # s = dft(img)
    s = fshift(s)
    pre_shifted = np.log(np.abs(s))
    return pre_shifted

def low_pass_filter(img, th=20):
    filter = np.fft.fft2(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if np.sqrt(((((filter.shape[0] - 1) // 2) - j) ** 2) + ((((filter.shape[1] - 1) // 2) - i) ** 2)) > th:
                filter[j, i] = 0
    filter = fshift(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

def high_pass_filter(img, th=30):
    filter = np.fft.fft2(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if np.sqrt(((((filter.shape[0] - 1) // 2) - j) ** 2) + ((((filter.shape[1] - 1) // 2) - i) ** 2)) < th:
                filter[j, i] = 0
    filter = fshift(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

def denoise1(img):
    return img

def denoise2(img):
    inner_ring = 41
    outer_ring = inner_ring + 2
    filter = np.fft.fft2(img)
    filter = fshift(filter)
    for j in range(filter.shape[0]):
        for i in range(filter.shape[1]):
            if np.sqrt(((((filter.shape[0] - 1) // 2) - j) ** 2) + ((((filter.shape[1] - 1) // 2) - i) ** 2)) < inner_ring\
                    or np.sqrt(((((filter.shape[0] - 1) // 2) - j) ** 2) + ((((filter.shape[1] - 1) // 2) - i) ** 2)) > outer_ring:
                filter[j, i] = 0
    filter = fshift(filter)
    filter = np.fft.ifft2(filter)
    return np.real(filter)

#################

if __name__ == '__main__':
    img = cv2.imread('./data/task2_sample.png', cv2.IMREAD_GRAYSCALE)
    cor1 = cv2.imread('./data/task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)
    cor2 = cv2.imread('./data/task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)

    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_pass_filter(img), 'Low-pass')
    drawFigure((2,7,3), high_pass_filter(img), 'High-pass')
    drawFigure((2,7,4), cor1, 'Noised')
    drawFigure((2,7,5), denoise1(cor1), 'Denoised')
    drawFigure((2,7,6), cor2, 'Noised')
    drawFigure((2,7,7), denoise2(cor2), 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(cor1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoise1(cor1)), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(cor2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoise2(cor2)), 'Spectrum')

    plt.show()