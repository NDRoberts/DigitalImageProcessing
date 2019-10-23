import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


VECTOR_LENGTH = 10000


def populate_data(target):

    images = {}
    images['entries'] = 0
    images['subjects'] = 0
    images[images['subjects']] = {}
    images[images['subjects']]['average'] = []

    img_num = 0

    for file in os.listdir(target):
        if img_num >= 5:
            images['subjects'] += 1
            images[images['subjects']] = {}
            img_num = 0
        path = target + file
        im = cv2.imread(path)
        images[images['subjects']][img_num] = im
        images['entries'] += 1
        img_num += 1
    images['subjects'] += 2
    return images



def vectorize(image):
    width, height, depth = image.shape[0:3]
    result = np.zeros((width * height), dtype=np.uint8)
    index = 0
    for m in range(width):
        for n in range(height):
            result[index] = image[m,n,0]
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


def subject_means(source):
    result = {}
    offset = 0
    subject = 1
    while offset < len(source):
        result[subject] = (source[offset] + source[offset + 1] + 
                           source[offset + 2] + source[offset + 3] + 
                           source[offset + 4]) / 5
        result[subject] = np.uint8(result[subject])
        subject += 1
        offset += 5
    return result


def minus_mean(src, mean):
    result = np.zeros((len(src),10000))
    for g in range(len(src)):
        result[g] = src[g] - mean
    return result


def top_ten_eigs(covariances):
    result = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    eigen = [0, 0]
    eigen[0], eigen[1] = np.linalg.eig(covariances)
    dictum = {0: [0, 0]}
    for p in range(len(eigen[0])):
        dictum[eigen[0][p]] = eigen[1][p]
    print(dictum.keys())
    ranked = (sorted(dictum))
    print(ranked)
    print("ZA BIGGESTO:", ranked[len(ranked)-1])
    print("The ranked list is", len(ranked), "poonits long.")
    print("Meanwhile, the covariance matrix was", len(covariances), "poonits wide.")
    for o in range(10):
        print(len(dictum))
        print(len(ranked))
        print(len(ranked)-o-1)
        result[o] = dictum[ranked[len(ranked)-o-1]]
    return result
    #max_vec = []
    #max_val = eigenvals[0]
    #max_i = 0
    #for i in range(len(eigenvals)):
    #    if eigenvals[i] > max_val:
    #        max_val = eigenvals[i]
    #        max_i = i
    #max_vec = eigenvects[max_i]
    #return (max_i, max_val, max_vec)


#------------------------------------------------------------------------------

print("Parsing face dataset...")

faces = populate_data('./Dataset/enrolling/')

print("Read complete.")
print("Subjects:", faces['subjects'])
print("Total images:", faces['entries'])

A = np.zeros((faces['entries'], 10000), dtype=np.uint8)

o = 0

for s in range(faces['subjects'] - 1):
    for p in range(5):
        A[o] = vectorize(faces[s][p])
        o += 1

v_mean = np.zeros(10000)
for i in range(A.shape[0]):
    v_mean += A[i]
v_mean = np.uint8(v_mean / A.shape[0])
cv2.imshow("Horrortown", reface(v_mean))
cv2.waitKey()
sub_mean = subject_means(A)
M = minus_mean(A, v_mean)
C = np.cov(M)

print(top_ten_eigs(C))