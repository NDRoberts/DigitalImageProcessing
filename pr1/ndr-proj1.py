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


def normaloid(vec):
    lomin = vec[0]
    lomax = vec[0]
    lenf = len(vec)
    ret = np.zeros(lenf, dtype = np.uint8)
    for a in range(lenf):
        if vec[a] < lomin:
            lomin = vec[a]
        elif vec[a] > lomax:
            lomax = vec[a]
    for b in range(lenf):
        ret[b] = (vec[b] - lomin) / (lomax - lomin)
    ret *= 255
    return ret


def envector(mat):
    if mat.shape[2] > 1:
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2YUV)
        mat = mat[:,:,0]
    indx = 0
    result = np.empty((1, VECTOR_LENGTH))
    for t in range(100):
        for u in range(100):
            result[0, indx] = mat[t,u]
            indx += 1
    return result


#------------------------------------------------------------------------------

print("Parsing face dataset...")
faces = populate_data('./Dataset/enrolling/')
print("Read complete.")
subjects = len(faces)
print("Subjects:", subjects)
print("Images per subject:", len(faces['ID01']))

# averages = subject_means(faces)
# print("Total averages produced:", len(averages))
# print("Length of each average:", averages['ID01'].shape[0])
# avg_mat = np.empty((subjects, VECTOR_LENGTH))
# i = 0
# for sub in averages:
#     avg_mat[i] = averages[sub]
#     i += 1

# o_mean = mean_vector(faces)

### Calculate big mats:
# big_A = np.zeros((subjects, VECTOR_LENGTH))
# big_min = 0
# big_max = 0

# for a in range(subjects):
#     big_A[a] = (avg_mat[a] - o_mean)
#     if min(big_A[a]) < big_min:
#         big_min = min(big_A[a])
#     if max(big_A[a]) > big_max:
#         big_max = max(big_A[a])

# print(big_min, big_max)
# big_A -= big_min
# big_A /= (big_max - big_min)
# big_A *= 255
# big_A = np.uint8(big_A)
# print(big_A)

row = 0
sub_avgs = np.zeros((subjects, VECTOR_LENGTH))
for subject in faces:
    average = np.zeros((1, VECTOR_LENGTH))
    for pic in faces[subject]:
        new_vec = envector(faces[subject][pic])
        average += new_vec
    for px in sub_avgs[row]:
        px = int(px / 5)
    row += 1
print("Subject averages matrix:", sub_avgs.shape)

overall_avg = np.zeros((1, VECTOR_LENGTH))
for r in range(subjects):
    overall_avg += sub_avgs[r]
for s in range(VECTOR_LENGTH):
    overall_avg[0][s] = int(overall_avg[0][s] / subjects)
print("Overall average vector:", overall_avg.shape)

big_A = sub_avgs - overall_avg
print("Big A matrix:", big_A.shape)

A = big_A.T
print("A dims:", A.shape)

P = np.dot(A, A.T)
print("P dims:", P.shape)

wt_A = np.dot(P.T, A)
print("wt_A dims:", wt_A.shape)

print("Parsing test dataset...")
test_set = populate_data('./Dataset/testing/')
print("Read complete.")

big_B = np.zeros(215, VECTOR_LENGTH)
for sub in test_set:
    for im in test_set[sub]:



# e_vals, P2 = np.linalg.eigh(np.dot(A.T, A))
# print("P2 size:", P2.shape)
#P = np.dot(big_A.T, P2)
#print("P size:", P.shape)
#wt_A = np.dot(P.T, big_A)
#print("wt_A size:", wt_A.shape)



# egg_on_face = np.dot(A, P2)

# for moop in egg_on_face.T:
#     cv2.imshow("Lumpy Gravy", reface(moop))
#     cv2.waitKey()

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


easy_mean, easy_eigen = cv2.PCACompute(avg_mat, mean=None)
abberfabs = np.uint8(easy_mean.reshape(100,100))
cv2.imshow('ngngng', abberfabs)
cv2.waitKey()
print("Autocalc mean:", easy_mean.shape)
print("Autocalc eigens:", easy_eigen.shape)
waytay = np.dot(easy_eigen.T, np.dot(A.T, A))
print("Weights or whatfuckever:", waytay.shape)
big_P = np.dot(A, easy_eigen)
print("big_P is", big_P.shape)

plt.subplot(221)
plt.title("Eig A")
plt.imshow(easy_eigen[0].reshape(100,100))
plt.subplot(222)
plt.title("Eig B")
plt.imshow(easy_eigen[1].reshape(100,100))
plt.subplot(223)
plt.title("Eig C")
plt.imshow(easy_eigen[2].reshape(100,100))
plt.subplot(224)
plt.title("Eig D")
plt.imshow(easy_eigen[3].reshape(100,100))
plt.show()

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
print("Size of those beez!?", len(big_B_array))

eudie = []
for p in range(len(waytay)):
    eudie.append(math.sqrt(sum((wt_B - waytay[p]))**2))
print(eudie)

# wate_bee = []
# for vec in big_B_array:
#     print(np.multiply(big_P.T, vec))
#     wate_bee.append(np.multiply(big_P.T, vec))
# print("Weights of B? ", wate_bee.shape)






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