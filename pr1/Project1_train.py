# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 22:10:20 2019

@author: jf
"""

from skimage.io import imread_collection
import numpy as np
from scipy import linalg
from utility import *

# Read in all Training images from images/
train_dir = "./images/enroll/*.bmp"

# Build multi-D collections for Training Data
train = imread_collection(train_dir)
X_arr = train.concatenate()

image_width = X_arr.shape[2]
image_height = X_arr.shape[1]
dims = (image_height, image_width)

vect_length = image_height*image_width
M = X_arr.shape[0] # number of training images
Mp = 44  # number of  person

X_flat = convert_to_vects(X_arr) # Convert images to vectors

person_dict = build_person_dict(train)
m = len(person_dict.keys())

print('     Constructing dictionary of {} persons on {} training images...'.format(m, M))

# 2. build avg vector for each person !!!!
#  create parallel list to hold img names
print('     Creating average vector of images from each face...')
avg_vects = np.empty((Mp, vect_length), dtype=X_arr[0].dtype)
avg_vects_names = []
row = 0
for key, value in person_dict.items():
     avg_vects_names.append(key)
            
     temp_vects = np.empty((len(value), vect_length), dtype=X_arr[0].dtype)
     for i in range(temp_vects.shape[0]):
         # print('{} : {}'.format(key,self.X_flat[value[i]]))
         temp_vects[i] = X_flat[value[i]]     
     # avg vectors for given image
     avg_vects[row] = temp_vects.mean(axis=0)  
     row += 1
print('     Done...')

# 3. Calc mean vector of all persons in training set
print('     Calculating mean of all vectors...')
mean_vect = avg_vects.mean(axis=0)
print('     Done...')

print('     Subtracting mean from all average images...')
X_flat_centered = np.empty(avg_vects.shape, dtype=X_arr[0].dtype)
for i in range(Mp):
    X_flat_centered[i] = center_vect(avg_vects[i], mean_vect)
X_flat_centered = X_flat_centered
print('     Done...')

print('     Computing eigenvalues and eigenvectors...')
cov_mat = np.cov(X_flat_centered.T)
eigenvals, eigenvects = linalg.eigh(cov_mat) # .eigh returns only real eigenvals
    
# reverse to put most contrib at beginning
ordered_eigenvals = eigenvals[::-1]
ordered_eigenvects = np.fliplr(eigenvects)

weights_of_flat_centered_arr = np.dot(X_flat_centered, ordered_eigenvects.T[:m].T)
np.save('wt_A.npy',weights_of_flat_centered_arr)
np.save('mean_vect.npy',mean_vect)


load_weight = np.load('wt_A.npy')
load_mean_vect = np.load('mean_vect.npy')