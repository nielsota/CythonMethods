cimport numpy as np
import numpy as np
import cython 
from scipy.linalg.cython_blas cimport ddot

cpdef double dot_product(double[::1] x1, double[::1] x2):
    """
    Cython method implementing the dot product
    
    Arguments
    ___________
    
    x1: 1 dimensional Fortran contigious array
    x2: 1 dimensional Fortran contigious array
    
    Return 
    ___________
    
    Dot product of vectors x1 and x2
    
    """
    
    if x1.shape[0] != x2.shape[0]:
        print("x1 and x2 must have same shape!")
    
    # define objects 
    cdef int N = x1.shape[0]
    cdef int incX1 = 1
    cdef int incX2 = 1
    
    # ddot is using variables outside its scope, so must use pointers
    return ddot(&N, &x1[0], &incX1, &x2[0], &incX2)
    