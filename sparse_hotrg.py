#!/anaconda/bin/python

import numpy as np
import sparten as st


def getU3d(tensor, nums, compute_sparse=False, return_sparse=True):
    ts = tensor.shape
    top = tensor.dot(tensor, ([1,2,3,4], [1,2,3,4]))
    bot = tensor.dot(tensor, ([1,2,3,5], [1,2,3,5]))
    Q = top.dot(bot, ([1,3], [1,3]))
    qs = Q.shape
    Q = (Q.transpose((0,2,1,3))).reshape((ts[0]*ts[0], ts[0]*ts[0]))
    Q = Q.to_csr()
    if (compute_sparse == True):
        evals, evecs = spsla.eigsh(Q, k=nums, which='LM', return_eigenvectors=True)
        print evals
    else:
        evals, evecs = np.linalg.eigh(Q.toarray())
        print evals
    idx = evals.argsort()[::-1]
    if (return_sparse == True):
        return st.tensor((evecs[:,idx])[:, :nums])
    else:
        return (evecs[:,idx])[:, :nums]
    
def getU2d(tensor, nums, compute_sparse=False, return_sparse=True):
    ts = tensor.shape
    top = tensor.dot(tensor, ([1,2], [1,2]))
    bot = tensor.dot(tensor, ([1,3], [1,3]))
    Q = top.dot(bot, ([1,3], [1,3]))
    qs = Q.shape
    Q = (Q.transpose((0,2,1,3))).reshape((ts[0]*ts[0], ts[0]*ts[0]))
    Q = Q.to_csr()
    if (compute_sparse == True):
        evals, evecs = spsla.eigsh(Q, k=nums, which='LM', return_eigenvectors=True)
        print evals
    else:
        evals, evecs = np.linalg.eigh(Q.toarray())
        print evals
    idx = evals.argsort()[::-1]
    if (return_sparse == True):
        return st.tensor((evecs[:,idx])[:, :nums])
    else:
        return (evecs[:,idx])[:, :nums]
    
def update2d(tensor, umat):
    ts = tensor.shape
    us = umat.shape
    us = (int(np.sqrt(us[0])), int(np.sqrt(us[0])), us[1])
    bot = tensor.dot(umat.reshape(us), ([0], [1]))
    top = tensor.dot(umat.reshape(us), ([1], [0]))
    want = bot.dot(top, ([3,1,0], [0,2,3]))
    return want.transpose((1,3,2,0))