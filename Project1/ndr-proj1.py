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
        print("This image has type", type(im[0,0,0]))
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
print("Subjects:", len(faces)+1)
print("Images per subject:", len(faces['ID01']))

averages = subject_means(faces)
print("Total averages produced:", len(averages)+1)
print("Length of each average:", averages['ID01'].shape[0])
print("First average, he looka like", averages['ID01'])

#for f in averages:
#    cv2.imshow("HOORS", reface(averages[f]))
#    cv2.waitKey()

o_mean = mean_vector(faces)
print("Showing da AVERGE", o_mean)
cv2.imshow("AVERGE", reface(o_mean))
cv2.waitKey()

eigenturkey = {}
for sub in averages:
    eigenturkey[sub] = (averages[sub] - o_mean)

for f in eigenturkey:
    cv2.imshow("HOORS", reface(eigenturkey[f]))
    cv2.waitKey()


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