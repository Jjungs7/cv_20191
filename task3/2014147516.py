import cv2
import numpy as np
import os

STUDENT_CODE = '2014147516'
FILE_NAME = 'output.txt'

if not os.path.exists(STUDENT_CODE):
    os.mkdir(STUDENT_CODE)

train_face_path = [f'face{("0" + str(idx))[-2:]}.pgm' for idx in range(1, 40)]
test_face_path = [f'test0{idx}.pgm' for idx in range(1, 6)]
images_train = np.array([cv2.imread(f'./faces_training/{t}', cv2.IMREAD_GRAYSCALE) for t in train_face_path])
images_test = np.array([cv2.imread(f'./faces_test/{t}', cv2.IMREAD_GRAYSCALE) for t in test_face_path])


def svd(img, percentage):
    mu_vector = np.mean(img)
    zero_mean = img - mu_vector
    unit_variance = zero_mean / (np.sum(zero_mean ** 2) / zero_mean.shape[1])

    #####
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    d = -1
    for i in range(len(s)):
        if (np.sum(s[:i]) / np.sum(s)) >= percentage:
            d = i
            break
    print(U.shape[0], U.shape[1])
    print(s.shape[0])
    print(Vt.shape[0], Vt.shape[1])

    return U, s, Vt, d


def main(args):
    percentage = float(args[len(args)-1])
    f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')
    images_train_2d = np.reshape(images_train, (39, -1))
    U, s, Vt, d = svd(images_train_2d, percentage)
    U2, _, Vt2, _ = svd(images_train[0], percentage)
    f.write('###########  STEP 1  ##########\n')
    f.write(f'Input Percentage: {percentage}\n')
    f.write(f'Selected Dimension: {d}\n')
    cv2.imwrite(f'./{STUDENT_CODE}/face01.pgm', np.dot(U, np.dot(np.diag(s)[:, :d], Vt[:d, :])))

    # step 2: reconstruct images

    #computed_images = np.array([cv2.imread(f'./{STUDENT_CODE}/{t}', cv2.IMREAD_GRAYSCALE) for t in train_face_path])
    #for i in range(len(computed_images)):
    #    f.write('\n###########  STEP 2  ##########\n')
    #    f.write(f'{("0" + str(i))[-2:]}: {np.mean((images_train[i] - computed_images[i]) ** 2)}\n')

    # step 3: face recognition
    #    f.write('\n###########  STEP 3  ##########\n')

    f.close()


if __name__=='__main__':
    import sys
    main(sys.argv)
