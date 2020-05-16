import os
import numpy as np
import scipy.io as sio

DEFAULT_FOLDER = 'D:/DataBase/HadamardMatrix/'

def load_hadamard_matrix(n, folder_name=DEFAULT_FOLDER):
    """
        Loads or creates Hadamard matrix with given input dimention n 
        and normalizes it such that H'H=I
    """
    
    fname = os.path.join(folder_name, 'H%d.mat' % n)
    if not os.path.exists(fname):
        if n == 1:
            H = np.array([1], dtype=np.float32)
        elif n % 2 != 0:
            H = None
        else:
            H = load_hadamard_matrix(n//2, folder_name)
            if H is not None:
                H = np.kron(np.array([[1, 1], [1, -1]]), H) / np.sqrt(2)
    else:
        data = sio.loadmat(fname)
        H = data['H'].astype(np.float32) / np.sqrt(n)
    
    return H
