#!/miniconda3/bin/python

import scipy.sparse as sps
import scipy.sparse.linalg as spsla
import numpy as np
from operator import mul
import copy
from functools import reduce

def prod(iterable):
    """
    Simple product function.
    
    Parameters
    ----------
    iterable : An iterable with things that can be multiplied
               together.
    
    Returns
    -------
    value : The product of the values.
    
    """
    return reduce(mul, iterable, 1)

# def new_tup(n, plist):
#     """
#     I only vaguely remember what I was thinking when I made this.
#     The idea is to convert every nonzero tensor index to a number
#     from the typical polynomial convention, and then reinvert from
#     polynomial number to the new index shape.  That is, for a tensor
#     of shape (N1, N2, N3) we reshape to (N1*N2*N3,) with:

#     (N1, N2, N3) --> N2*N3*i + N3*j + k

#     If (M1, M2) is a new shape such that M1*M2 == N1*N2*N3 then invert
    
#     N2*N3*i + N3*j + k == M2*i + j --> (M1, M2).

    
#     Parameters
#     ----------
#     n     : The value of the flattened tensor index.
#     plist : I think this is the a list of products of the
#             new tensor shape tuple.
#     Returns
#     -------
#     tensor_tuple : The new tuple of tensor indices according to the
#                    new input shape.
#     """
#     ijk = list()
#     a, xx = divmod(n, plist[0])
#     ijk.append(a)
#     for x in plist[1:]:
#         a, xx = divmod(xx, x)
#         ijk.append(a)
#     return tuple(ijk)


# def list_prod(list1, list2):
#     a = len(list1)
#     assert a == len(list2)
#     want = [0]*a
#     for i in range(a):
#         want[i] = list1[i]*list2[i]
#     return want



class tensor:
    """
    A sparse tensor class.
    """
    
    def __init__(self, array=None, tol=1e-14):
        """
        Initialize the sparse tensor.
        
        Parameters
        ----------
        array : The array to be converted to a sparse tensor.
        tol   : The tolerance below which values are ignored
                and set to zero when creating the tensor.
        """
        # Check what the input is and make the
        # set-up accordingly
        if (type(array) == np.ndarray):
            truth = np.abs(array) > tol
            self.idx = np.argwhere(truth)
            self.vals = array[truth]
            self.shape = array.shape
        elif (type(array) == tuple):
            self.idx = array[0]
            self.vals = array[1]
            self.shape = array[2]
        elif (array == None):
            self.shape = tuple()
            self.idx = np.array()
            self.vals = np.array()
        else:
            raise TypeError("Array from which to build sparse tensor not valid.")

    def __mul__(self, other):
        """
        Multiplication.  Right now only for scalars.
        """
        if isinstance(other, (int, float, complex)):
            return tensor((self.idx, self.vals*other, self.shape))
        else:
            raise ValueError("Don't know how to multiply by that.")

    def norm(self):
        """
        Computes the norm of the tensor ASSUMING IT IS REAL!!!
        """
        return np.linalg.norm(self.vals)

    def trace(self, contracted_indices):
        """
        Computes the trace over two indices.
        """
        assert len(contracted_indices) == 2
        cidx = list(contracted_indices)
        assert self.shape[cidx[0]] == self.shape[cidx[1]]
        one = tensor(np.identity(self.shape[cidx[0]]))
        want = self.dot(one, (cidx, [0,1]))
        return want


    def reshape(self, size_tuple):
        """
        Reshapes the sparse tensor into another tensor
        with the same total size.
        
        Parameters
        ----------
        size_tuple : The new tuple shape.
        
        Returns
        -------
        new_array : A new array with the shape `size_tuple' created
                    from the original tensor.
        
        """
        assert prod(self.shape) == prod(size_tuple)
        
        ts = self.shape
        ps = size_tuple
        ts = np.array([prod(ts[i+1:]) for i in range(len(ts)-1)] + [1])
        ps = np.array([prod(ps[i+1:]) for i in range(len(ps)-1)] + [1])
        flat_vals = np.sum(ts*self.idx, axis=1)
        new_idx = np.zeros((len(self.vals), len(size_tuple)))
        rem = np.zeros(len(self.vals))
        np.divmod(flat_vals, ps[0], out=(new_idx[:,0], rem))
        for i in range(1, len(size_tuple)):
            np.divmod(rem, ps[i], out=(new_idx[:, i], rem))
        return tensor((new_idx.astype(int), self.vals, size_tuple))
                   
        
    def transpose(self, new_order):
        """
        Transposes the indices of the tensor.
        
        Parameters
        ----------
        new_order : The new order of the tensor indices.
        
        Returns
        -------
        new_array : A new array with the same size as the original
                    array however with transposed indices according
                    to `new_order'.
                    
        """
        assert len(new_order) == len(self.shape)

        new_shape = tuple([self.shape[i] for i in new_order])
        return tensor((self.idx[:, new_order], self.vals, new_shape))
        
    
    def dot(self, tensor2, contracted_indices):
        """
        Contracts two sparse tensors together accoding to `contracted_indices'.
        
        Parameters
        ----------
        tensor2            : The second tensor that's contracted with this
                             tensor.
        contracted_indices : The indices over which the two tensors will be
                             contracted.
        
        Returns
        -------
        new_array : A new tensor built by contracting tensor2 and this tensor
                    over their common indices.
                    
        """
        assert (len(contracted_indices) == 2)

        # get the tensor shapes
        ts1 = self.shape
        ts2 = tensor2.shape

        # get the contracted indices
        ax1 = contracted_indices[0] # this is a tuple of indices for self
        ax2 = contracted_indices[1] # ditto for tensor2

        # build the transposed tuples
        idx1 = list(range(len(ts1)))
        idx2 = list(range(len(ts2)))
        for n in ax1:
            idx1.remove(n)
        for n in ax2:
            idx2.remove(n)
        id1f = tuple(list(idx1) + list(ax1))
        id2f = tuple(list(ax2) + list(idx2))

        # transpose the input tensors to prepare
        # for matrix multiplication
        tleft = self.transpose(id1f)
        tright = tensor2.transpose(id2f)

        # now sperate and reshape into two index objects
        ts1 = tleft.shape
        ts2 = tright.shape
        left = ts1[:len(idx1)]
        right = ts2[len(ax2):]
        final = tuple(list(left) + list(right))
        assert (len(ax2) == (len(ts1)-len(idx1)))
        tleft = tleft.reshape((prod(ts1[:len(idx1)]), prod(ts1[len(idx1):])))
        tright = tright.reshape((prod(ts2[:len(ax2)]), prod(ts2[len(ax2):])))
        # make sparse matricies in CSR format
        tlidx = tleft.idx
        tleft = sps.csr_matrix((tleft.vals, (tlidx[:,0], tlidx[:,1])), shape=tleft.shape)
        tridx = tright.idx
        tright = sps.csr_matrix((tright.vals, (tridx[:,0], tridx[:,1])), shape=tright.shape)

        # dot and reshape into final tensor
        tleft = tleft.dot(tright)
        del tright
        new_idx = np.transpose(np.vstack(tleft.nonzero()))
        new_vals = tleft.data
        return tensor((new_idx, new_vals, tleft.shape)).reshape(final)
    
    def to_csr(self,):
        """
        Convert sparten tensor to scipy.csr type
        
        Returns
        -------
        new_matrix : A copy of the matrix in csr format.
        """
        assert (len(self.shape) == 2)
        tidx = self.idx
        return sps.csr_matrix((self.vals, (tidx[:,0], tidx[:,1])), shape=self.shape)
        
    
        
    
# def tensordot(tensor1, tensor2, contracted_indices):
#     """
#     Contracts two sparse tensors together accoding to `contracted_indices'.

#     Parameters
#     ----------
#     tensor1            : The first tensor to contract.
#     tensor2            : The second tensor that's contracted.
#     contracted_indices : The indices over which the two tensors will be
#                          contracted.

#     Returns
#     -------
#     new_array : A new tensor built by contracting tensor2 and this tensor
#                 over their common indices.
                    
#     """
#     assert (len(contracted_indices) == 2)
    
#     # get the tensor shapes
#     ts1 = tensor1.shape
#     ts2 = tensor2.shape
    
#     # get the contracted indices
#     ax1 = contracted_indices[0] # this is a tuple of indices for tensor1
#     ax2 = contracted_indices[1] # ditto for tensor2
    
#     # build the transposed tuples
#     idx1 = range(len(ts1))
#     idx2 = range(len(ts2))
#     for n in ax1:
#         idx1.remove(n)
#     for n in ax2:
#         idx2.remove(n)
#     id1f = tuple(list(idx1) + list(ax1))
#     id2f = tuple(list(ax2) + list(idx2))
    
#     # transpose the input tensors to prepare
#     # for matrix multiplication
#     tleft = tensor1.transpose(id1f)
#     tright = tensor2.transpose(id2f)
    
#     # now sperate and reshape into two-index objects
#     ts1 = tleft.shape
#     ts2 = tright.shape
#     left = ts1[:len(idx1)]
#     right = ts2[len(ax2):]
#     final = tuple(list(left) + list(right))
#     assert (len(ax2) == (len(ts1)-len(idx1)))
#     tleft = tleft.reshape((prod(ts1[:len(idx1)]), prod(ts1[len(idx1):])))
#     tright = tright.reshape((prod(ts2[:len(ax2)]), prod(ts2[len(ax2):])))
    
#     # make sparse matricies in CSR format
#     tlidx = np.asarray(tleft.keys())
#     tleft = sps.csr_matrix((tleft.values(), (tlidx[:,0], tlidx[:,1])), shape=tleft.shape)
#     tridx = np.asarray(tright.keys())
#     tright = sps.csr_matrix((tright.values(), (tridx[:,0], tridx[:,1])), shape=tright.shape)
    
#     # dot and reshape into final tensor
#     tleft = tleft.dot(tright)
#     del tright
#     tleft = tleft.todok(copy=False)
#     tleft = tensor(tleft).reshape(final)
#     return tleft
    
    
