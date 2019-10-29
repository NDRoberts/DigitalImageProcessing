import cv2
import math
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
    side = int(math.sqrt(VECTOR_LENGTH))
    z = 0
    result = np.zeros((side, side), dtype=np.uint8)
    for j in range(side):
        for k in range(side):
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
subjects = len(faces)
print("Subjects:", subjects)
print("Images per subject:", len(faces['ID01']))

averages = subject_means(faces)
print("Total averages produced:", len(averages))
print("Length of each average:", averages['ID01'].shape[0])
avg_mat = np.empty((subjects, VECTOR_LENGTH))
i = 0
for sub in averages:
    avg_mat[i] = averages[sub]
    i += 1

o_mean = mean_vector(faces)

### Calculate big mats:
big_A = np.zeros((subjects, VECTOR_LENGTH))
big_min = 0
big_max = 0

for a in range(subjects):
    big_A[a] = (avg_mat[a] - o_mean)
    if min(big_A[a]) < big_min:
        big_min = min(big_A[a])
    if max(big_A[a]) > big_max:
        big_max = max(big_A[a])

print(big_min, big_max)


print("big_A size:", big_A.shape)
print("big_A transpose size:", big_A.T.shape)

A = big_A.T

e_vals, P2 = np.linalg.eigh(np.dot(A.T, A))
print("P2 size:", P2.shape)
#P = np.dot(big_A.T, P2)
#print("P size:", P.shape)
#wt_A = np.dot(P.T, big_A)
#print("wt_A size:", wt_A.shape)
wt_A = np.dot(A, np.dot(A.T, A))

egg_on_face = np.dot(A, P2)

for moop in egg_on_face.T:
    cv2.imshow("Lumpy Gravy", reface(moop))
    cv2.waitKey()

# np.save("C:\\Code\\big_A.npy", big_A)
# np.save("C:\\Code\\P2_ata.npy", P2_ata)
# np.save("C:\\Code\\P2.npy", P2)
# np.save("C:\\Code\\P.npy", P)
# np.save("C:\\Code\\wt_A.npy", wt_A)


### Load big mats from files:
# big_A = np.load("C:\\Code\\big_A.npy")
# print("big_A Size:", big_A.shape)
# P2_ata = np.load("C:\\Code\\P2_ata.npy")
# print("P2_ata Size:", P2_ata.shape)
# P2 = np.load("C:\\Code\\P2.npy")
# print("P2 Size:", P2.shape)
# P = np.load("C:\\Code\\P.npy")
# print("P Size:", P.shape)
# wt_A = np.load("C:\\Code\\wt_A.npy")

# eigenface = np.dot(big_A, P2)

o = 0
test_faces = populate_data('./Dataset/testing/')
# test_array = np.empty((subjects, VECTOR_LENGTH))
# imvec = np.zeros(VECTOR_LENGTH, dtype=np.uint8)
big_B_array = []
for sub in test_faces:
    for img in test_faces[sub]:
        #imvec = vectorize(test_faces[sub][img])
        #test_array[o] = imvec
        big_B_array.append(vectorize(test_faces[sub][img]) - o_mean)
# big_B = test_array[0] - o_mean
# print("Size of big_B:", big_B.shape)

wt_B = np.multiply(P.T, big_B_array[0])

euclist = []
for subj in test_faces:
    for im in test_faces[subj]:
        print("Test Photo:", subj, im)
        vector_B = vectorize(test_faces[subj][im]) - o_mean
        weight_B = np.multiply(P.T, vector_B)
        euclist.append(math.sqrt(sum((weight_B - wt_A[0]))))






print("Size of wt_A:", wt_A.shape)
print("Size of wt_B:", wt_B.shape)

print("I will examine wt_B[0].")
e_distance = []
for p in range(subjects):
    e_distance.append(math.sqrt(sum((wt_B[0] - wt_A[p])**2)))

mindex = 0
for q in range(len(e_distance)):
    if e_distance[q] < e_distance[mindex]:
        mindex = q
print("I think wt_B[0] belongs to subject", mindex)




# covars = np.cov(big_A.T)
# eigenvals, eigenvects = np.linalg.eigh(covars)
# ordered_eigenvals = eigenvals[::-1]
# ordered_eigenvects = np.fliplr(eigenvects)

# weight_a = np.dot(big_A, ordered_eigenvects.T[:subjects].T)


# np.save('C:\\Code\\wt_a.npy', wt_A)
# np.save('C:\\Code\\mean_vect.npy', o_mean)

# plt.plot(wt_A)
# plt.show()