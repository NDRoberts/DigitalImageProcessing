import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


VECTOR_LENGTH = 10000


def populate_data(target):

    images = {}
    images['entries'] = 0
    images['subjects'] = 0

    for file in os.listdir(target):
        if (images['entries'] % 5) == 0:
            images['subjects'] += 1
            images[images['subjects']] = {}
        path = target + file
        im = cv2.imread(path)
        images[images['subjects']][img_num] = im
        images['entries'] += 1
    
    return images



def vectorize(image):
    width, height, depth = image.shape[0:3]
    result = np.zeros((width * height))
    index = 0
    for m in range(width):
        for n in range(height):
            result[index] = image[m,n,0]
            index += 1
    return result


print("Parsing face dataset...")

faces = populate_data('./Dataset/enrolling/')

print("Read complete.")
print("Subjects:", faces['subjects'])
print("Total images:", faces['entries'])

A = np.zeros((faces['entries'], 10000))

o = 0

for s in range(faces['subjects']):
    for p in range(5):
        print(faces[s][p])
        #A[o] = vectorize(faces[s][p])
        #o += 1
