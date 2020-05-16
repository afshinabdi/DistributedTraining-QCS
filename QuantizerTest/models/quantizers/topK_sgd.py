"""
    Implementation of the paper 'Sparsified SGD with Memory'
    This is mainly based on the code available at 'https://github.com/epfml/sparsifiedSGD'
    especially the file memory.py
"""

import numpy as np

class topk_quantizer:
    def __init__(self, k):
        self._k = k
        self._residue = 0

    def quantize(self, X, reconstructed=True):
        """
            Top-K SGD sparsification with memory
            Parameters:
                g (np:ndarray) : input gradient
                residue (np:ndarray) : residue of the same shape as g
                k (int) : number of elements to keep
        """

        self._residue += X
        self._k = min(X.size, self._k)
        indices = np.argpartition(np.abs(self._residue.ravel()), -self._k)[-self._k:]
        indices = np.unravel_index(indices, X.shape)

        Xh = np.zeros_like(self._residue)
        Xh[indices] = self._residue[indices]
        self._residue[indices] = 0

        if reconstructed:
            return Xh
        else:
            return indices, Xh[indices]

    def reset(self):
        self._residue = 0
