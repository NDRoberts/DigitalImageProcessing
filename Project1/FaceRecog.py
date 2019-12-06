import math
import cv2
import numpy as np

from matplotlib import pyplot as plt
from skimage.io import imread_collection


def vector_average(matrix):
    ''' Compute the average vector from a matrix of vectors '''
    inputs = matrix.shape[0]
    vec_len = matrix.shape[1]
    result = np.zeros((vec_len))
    for r in range(inputs):
        result += matrix[r]
    for v in range(vec_len):
        result[v] /= inputs
    result = np.uint8(result)
    return result


def standardize(src):
    ''' Compute the normalized values of a vector to within (0, 255) '''
    result = np.zeros(src.shape, dtype=src.dtype)
    src_squared = []
    for val in src:
        src_squared.append(val**2)
    mu = sum(src) / len(src)
    mu_squared = sum(src_squared) / len(src_squared)
    variance = mu_squared - mu
    sigma = math.sqrt(variance)
    for k in range(len(src)):
        result[k] = ((src[k] - mu) / sigma) * 255
    return result


# Read in images; should be 43 subjects, 100x100 image size
TRAINING_PATH = "./Dataset/enrolling/*.bmp"
TRAINING_DATA = imread_collection(TRAINING_PATH)

# Array of all images together; should be 215x100x100
X_array = TRAINING_DATA.concatenate()

TOTAL_IMAGES = X_array.shape[0]
IMG_HEIGHT = X_array.shape[1]
IMG_WIDTH = X_array.shape[2]
DIMENSIONS = (IMG_HEIGHT, IMG_WIDTH)
VECTOR_LENGTH = IMG_HEIGHT * IMG_WIDTH
NUM_SUBJECTS = 43

# Flatten images into vectors; vector array should be 215x10000
img_vectors = np.empty((TOTAL_IMAGES, VECTOR_LENGTH), dtype=X_array.dtype)
for im in range(TOTAL_IMAGES):
    img_vectors[im] = np.ravel(X_array[im])

# "X" Average each subject; average array should be 43x10000, overall average 1x10000
averages = np.empty((NUM_SUBJECTS, VECTOR_LENGTH), dtype=np.float32)
offset = 0
for a in range(NUM_SUBJECTS):
    n = 5 * offset
    averages[a] = vector_average(img_vectors[n+0:n+5])
    offset += 1
overall_average = vector_average(averages)

print("Overall average:", overall_average)
print("Average of Subject 1:", averages[0])
print("Average of Subject 10:", averages[9])

# "A": subtract overall average from single averages; array should be 43x10000
avg_reduced = np.zeros(averages.shape, dtype=averages.dtype)
for b in range(NUM_SUBJECTS):
    avg_reduced[b] = np.clip((averages[b] - overall_average), 0, 255)
avg_reduced = np.uint8(avg_reduced)

print("Centered average of Subject 1:", avg_reduced[0])
print("Centered average of Subject 10:", avg_reduced[9])

criggity_covars = np.cov(avg_reduced)

shanks, shonks = np.linalg.eig(criggity_covars)



# Dot product of A*A'; should be 43x43
ata = np.matmul(avg_reduced, avg_reduced.T)

# "P2": Get eigenvalues/vector of A*A'; vector array should be 43x43
eigenvalues, eigenvectors = np.linalg.eig(ata)
#np.save('eigenvectors2t.npy', eigenvectors)

# "wt_A": Weight of vectors projected into eigenspace as P2*(A*A'); should be 43x43 
eigenweight = np.matmul(eigenvectors, ata)
#np.save('eigenweight2t.npy', eigenweight)

# "P": Centered averages times eigenvectors; should be 10000x43
eigenfaces = np.matmul(avg_reduced.T, eigenvectors)
# Flip eigenfaces back to 43x10000
eigenfaces = np.uint8(eigenfaces.T)
# np.save('eigenfaces2t.npy', eigenfaces)
print("Eigenface vector for subject 1:", eigenfaces[0])
print("Eigenface vector for subject 10:", eigenfaces[9])

# eigenvectors = np.load('eigenvectors2t.npy')
# eigenweight = np.load('eigenweight2t.npy')
# eigenfaces = np.load('eigenfaces2t.npy')

eigface1 = np.reshape(np.uint8(eigenfaces[0]), (100, 100))
eigface2 = np.reshape(np.uint8(eigenfaces[9]), (100, 100))
# cv2.imshow("Eigenface", eigface1)
# cv2.imshow("Other Eigenface", eigface2)
# cv2.waitKey()

# plt.subplot(221)
# plt.title("Average: Subject 1")
# plt.imshow(np.reshape(averages[0], (100, 100)))
# plt.subplot(222)
# plt.title("Average: Overall")
# plt.imshow(np.reshape(overall_average, (100, 100)))
# plt.subplot(223)
# plt.title("Subject 1 Centered")
# plt.imshow(np.reshape(avg_reduced[0], (100, 100)))
# plt.subplot(224)
# plt.title("Eigenface")
# plt.imshow(np.reshape(eigface1, (100, 100)))
# plt.show()

plt.subplot(131)
plt.title("Eigenface 1")
plt.imshow(np.reshape(np.uint8(eigenfaces[0]), (100, 100)))
plt.subplot(132)
plt.title("Eigenface 10")
plt.imshow(np.reshape(np.uint8(eigenfaces[10]), (100, 100)))
plt.subplot(133)
plt.title("Eigenface 20")
plt.imshow(np.reshape(np.uint8(eigenfaces[20]), (100, 100)))
plt.show()

plt.subplot(321)
plt.title("Subject 1")
plt.plot(eigenfaces[0])
plt.subplot(322)
plt.title("Subject 10")
plt.plot(eigenfaces[10])
plt.subplot(323)
plt.plot(averages[0])
plt.subplot(324)
plt.plot(averages[10])
plt.subplot(325)
plt.hist(averages[0], bins = 256)
plt.subplot(326)
plt.hist(averages[10], bins = 256)
plt.show()

print("Averages, Face 1:", averages[0][0:10])
print("Eigenvector, Face 1:", eigenvectors[0][0:10])
print("Eigenface, Face 1:", eigenfaces[0][0:10])

# for face in eigenfaces:
#     cv2.imshow("Eigenface", np.reshape(face, (100,100)))
#     cv2.waitKey()