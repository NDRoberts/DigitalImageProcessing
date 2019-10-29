import numpy as np


def convert_to_vect(mat_):

        mat = mat_.copy()
        vect = np.ravel(mat)
        return vect
            
def convert_to_vects(arr_):
        '''
        flattens all images in a numpy array into vectors

        Input:
                arr_: numpy array 
        '''
        print('     Converting 2D images to vectors...')
        
        arr = arr_.copy()
        vect_length = arr.shape[1]*arr.shape[2]
        M = arr.shape[0]         # number of training images
      
        if M == 1: # vectorize single image
            return np.ravel(arr)
        else:
            arr_flat = np.empty((M, vect_length), dtype=arr[0].dtype)
            for i in range(M):
                arr_flat[i] = convert_to_vect(arr[i])
            return arr_flat
        

def build_person_dict(X):
        '''
        Assumes that filenames are in the format:
            ID<2 digit person id>_<3 digit image number>.bmp
            example: ID00_006.bmp

        Input:
                X: skimage imread_collection
                collection of 2d images to build filename associations
        '''
        # extract filenames in order to build img averages
        X_filenames = X.files
        
        # Build dictionary of indices for each person
        person_dict = dict()
        for i in range(len(X_filenames)):
            person = X_filenames[i][0:-8]
            if person not in person_dict:
                person_dict[person] = []
            person_dict[person].append(i)
            
        return person_dict

def center_vect(vect,mean_vect):
        centered = vect - mean_vect
        centered = centered.clip(min=0)
        
        return centered       
        
