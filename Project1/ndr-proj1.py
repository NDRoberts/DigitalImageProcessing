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


def mean_vector(matrix):
    l = len(matrix)
    result = np.zeros(len(matrix[0]))
    for i in range(l):
        result += matrix[i]
    result /= l
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
print("Subjects:", len(faces)+1)
print("Images per subject:", len(faces['ID01']))


'''
print("Read complete.")
print("Subjects:", faces['subjects'])
print("Total images:", faces['entries'])

avg_faces = subject_means(faces)

overall_avg = mean_vector(faces)

big_A = np.empty((faces['subjects'], VECTOR_LENGTH))

for t in range(faces['subjects']):
    big_A[t] = avg_faces[t] - overall_avg
'''