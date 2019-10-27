import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


VECTOR_LENGTH = 10000


def populate_data(target):
    raw_data = dict()
    for file in os.listdir(target):
        fname = file[0:8]
        subject = file[0:4]
        img_num = file[5:8]
        path = target + file
        im = cv2.imread(path)
        if subject not in raw_data:
            raw_data[subject] = {}
        raw_data[subject][img_num] = im
    return raw_data


def vectorize(image):
    width, height, depth = image.shape[0:3]
    result = np.zeros((width * height), dtype=np.uint8)
    index = 0
    yek = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    for m in range(width):
        for n in range(height):
            result[index] = yek[m,n,0]
            index += 1
    return result


def reface(vector):
    z = 0
    result = np.zeros((100,100), dtype=np.uint8)
    for j in range(100):
        for k in range(100):
            result[j,k] = vector[z]
            z += 1
    return result


def mean_vector(source):
    result = np.zeros(VECTOR_LENGTH)
    i = 0
    for sub in source:
        for img in source[sub]:
            result += vectorize(source[sub][img])
            i += 1
    result = np.uint8(result / i)
    return result


def subject_means(source):
    result = {}
    avec = np.zeros(VECTOR_LENGTH, dtype=np.uint8)
    for sub in source:
        for img in source[sub]:
            avec += vectorize(source[sub][img])
        avec = np.uint8(avec / len(sub))
        result[sub] = avec
    return result


def best_eigs(covariances):
    result = []
    eigen = [0, 0]
    eigen[0], eigen[1] = np.linalg.eig(covariances)
    for e in range(len(eigen[0])):
        if eigen[0][e] >= 0:
            result.append(eigen[1][e])
    return result


#------------------------------------------------------------------------------

print("Parsing face dataset...")

faces = populate_data('./Dataset/enrolling/')
print("Read complete.")
subjects = len(faces)+1
print("Subjects:", subjects)
print("Images per subject:", len(faces['ID01']))

averages = subject_means(faces)
print("Total averages produced:", len(averages)+1)
print("Length of each average:", averages['ID01'].shape[0])
avg_mat = np.empty((subjects, VECTOR_LENGTH))
i = 0
for sub in averages:
    avg_mat[i] = averages[sub]
    i += 1

o_mean = mean_vector(faces)

big_A = np.zeros((subjects, VECTOR_LENGTH))
for a in range(subjects):
    big_A[a] = np.clip((avg_mat[a] - o_mean),0,255)

# a_trans_a = np.dot(big_A.T, big_A)
# e_vals, e_vects = np.linalg.eigh(a_trans_a)
# wt_a = np.dot(e_vects.T, a_trans_a)

# eigenface = np.dot(big_A, e_vects)

covars = np.cov(big_A.T)
eigenvals, eigenvects = np.linalg.eigh(covars)
ordered_eigenvals = eigenvals[::-1]
ordered_eigenvects = np.fliplr(eigenvects)

weight_a = np.dot(big_A, ordered_eigenvects.T[:subjects].T)

np.save('wt_a.npy', weight_a)
np.save('mean_vect.npy', o_mean)

plt.plot(weight_a)
plt.show()