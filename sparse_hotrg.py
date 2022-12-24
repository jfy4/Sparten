#!/anaconda/bin/python

import numpy as np
import sparten as st
import scipy.sparse.linalg as spsla


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
#         print evals
    else:
        evals, evecs = np.linalg.eigh(Q.toarray())
#         print evals
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
#         print evals
    else:
        evals, evecs = np.linalg.eigh(Q.toarray())
#         print evals
    idx = evals.argsort()[::-1]
    if (return_sparse == True):
        return st.tensor((evecs[:,idx])[:, :nums])
    else:
        return (evecs[:,idx])[:, :nums]

def exact_contract(tensor1, tensor2):
    ts1 = tensor1.shape
    ts2 = tensor2.shape
    final = tensor1.dot(tensor2, ([3], [2]))
    final = final.transpose((0,3, 1,4, 2, 5))
    final = final.reshape((ts1[0]*ts2[0], ts1[1]*ts2[1], ts1[2], ts2[3]))
    return final


def update2d(tensor, umat):
#     ts = tensor.shape
    us = umat.shape
    us = (int(np.sqrt(us[0])), int(np.sqrt(us[0])), us[1])
    bot = tensor.dot(umat.reshape(us), ([0], [1]))
    top = tensor.dot(umat.reshape(us), ([1], [0]))
    want = bot.dot(top, ([3,1,0], [0,2,3]))
    return want.transpose((1,3,2,0))

def traceupdate2d(tensor):
    ts = tensor.shape
    traced = tensor.dot(tensor, ([2,3], [3,2]))
    traced = traced.transpose((0,2,1,3))
    return traced.reshape((ts[0]**2, ts[1]**2))

def update3d(tensor, umat):
    # tensor is assumed left right front back top bottom
    us = umat.shape
    us = (int(np.sqrt(us[0])), int(np.sqrt(us[0])), us[1])
    
    bot = tensor.dot(umat.reshape(us), ([0], [1]))
    bot = bot.dot(umat.reshape(us), ([1], [1]))
    
    top = tensor.dot(umat.reshape(us), ([1], [0]))
    top = top.dot(umat.reshape(us), ([2], [0]))
    
    want = bot.dot(top, ([0,1,2,4,6], [4,6,3,0,1]))
    return want.transpose((1,4,2,5,3,0))
