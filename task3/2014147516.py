import cv2
import numpy as np
import os

STUDENT_CODE = '2014147516'
FILE_NAME = 'output.txt'

if not os.path.exists(STUDENT_CODE):
    os.mkdir(STUDENT_CODE)

train_file_names = [f'face{("0" + str(idx))[-2:]}.pgm' for idx in range(1, 40)]
test_file_name = [f'test0{idx}.pgm' for idx in range(1, 6)]
images_train = np.array([cv2.imread(f'./faces_training/{t}', cv2.IMREAD_GRAYSCALE) for t in train_file_names], dtype=np.int)
image_h = images_train[0].shape[0]
image_w = images_train[0].shape[1]


def svd(img, percentage):
    # mu_vector = np.mean(img)
    # zero_mean = img - mu_vector
    # unit_variance = zero_mean / (np.sum(zero_mean ** 2) / zero_mean.shape[1])

    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    s2 = np.square(s)
    d = -1
    for i in range(len(s2) + 1):
        if (np.sum(s2[:i]) / np.sum(s2)) >= percentage:
            d = i
            break
    return U, s, Vt, d


def main(args):
    percentage = float(args[len(args)-1])

    f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')

    """
    ###### STEP 1 #######
    Reshape image to N*1 vector
    and connect all vectors to form N*n (where n is num of faces)
    """
    f.write('##########  STEP 1  ##########\n')
    images_train_2d = np.zeros((images_train[0].shape[0] * images_train[0].shape[1], 39), dtype=np.int)
    for idx, image in enumerate(images_train):
        for i in range(image.shape[0]):
            images_train_2d[image.shape[1] * i: image.shape[1] * (i + 1), idx] = np.array(image[i, :], dtype=np.int)

    means = np.array([], dtype=np.float)
    for i in range(images_train_2d.shape[0]):
        mean = 0
        for j in range(39):
            mean += images_train_2d[i, j]
        mean = mean / 39
        for j in range(images_train_2d.shape[1]):
            images_train_2d[i, j] = images_train_2d[i, j] - mean
        means = np.append(means, mean)

    U, s, Vt, d = svd(images_train_2d, percentage)
    eigen_face = U[:, :d]

    f.write(f'Input Percentage: {percentage}\n')
    f.write(f'Selected Dimension: {d}\n')
    result = np.dot(U[:, :d], np.dot(np.diag(s)[:d, :d], Vt[:d, :]))
    for i in range(images_train_2d.shape[0]):
        for j in range(39):
            result[i, j] += means[i]

    '''
    result = np.zeros((image_h * image_w, 39))
    for idx, image in enumerate(images_train):
        image = np.reshape(image, image.shape[0] * image.shape[1]) - means
        res = np.zeros(image_h * image_w)
        for i in range(d):
            res += np.dot(eigen_face[:, i], np.dot(eigen_face[:, i].T, image))
        result[:, idx] = means + res
    '''

    for idx, file_name in enumerate(train_file_names):
        cv2.imwrite(f'./{STUDENT_CODE}/{file_name}', np.reshape(result[:, idx], (image_h, image_w)))


    """
    ###### STEP 2 #######
    compute reconstruction error between original images and reconstructed images
    """
    f.write('\n##########  STEP 2  ##########\n')
    results = []
    computed_images = np.array([cv2.imread(f'./{STUDENT_CODE}/{t}', cv2.IMREAD_GRAYSCALE) for t in train_file_names], dtype=np.int)
    for i in range(len(computed_images)):
        difference = np.mean((images_train[i] - computed_images[i]) ** 2)
        results.append(difference)

    f.write('Reconstruction error\n')
    f.write(f'average : {"%.4f" % np.average(results)}\n')
    for i in range(len(results)):
        f.write(f'{("0" + str(i+1))[-2:]}: {"%.4f" % results[i]}\n')

    """
    ###### STEP 3 ######
    Face recognition using l2 distance
    """
    f.write('\n##########  STEP 3  ##########\n')
    images_test = np.array([cv2.imread(f'./faces_test/{t}', cv2.IMREAD_GRAYSCALE) for t in test_file_name], dtype=np.int)
    for idx, image in enumerate(images_test):
        image = np.reshape(image, image.shape[0] * image.shape[1]) - means
        res = np.zeros(image_h * image_w)
        for i in range(d):
            res += np.dot(eigen_face[:, i], np.dot(eigen_face[:, i].T, image))
        final_img = np.reshape(means + res, (image_h, image_w))
        results = []
        for image_comp in computed_images:
            results.append(np.sqrt(np.sum((final_img - image_comp) ** 2)))
        f.write(f'test0{idx + 1}.pgm ==> ')
        f.write(f'face{("0" + str(results.index(min(results)) + 1))[-2:]}.pgm\n')

    f.close()


if __name__ == '__main__':
    import sys
    main(sys.argv)
