import numpy as np
from scipy.linalg import inv, svd
from tqdm import tqdm_notebook as tqdm

### Helper functions

# Compute null space
def null_space(A, rcond=None):
    """
    Compute null spavce of matrix XProjection on half space defined by {v| <v,w> = c}
    Arguments:
        A {numpy.ndarray} -- matrix whose null space is desired
        rcond {float} -- intercept
    Returns:
        Q {numpy.ndarray} -- matrix whose (rows?) span null space of A
    """
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

### End Helper Functions

# Exercise 1: Alternating projection for subspaces
def altproj(A, B, v0, n):
    """
    Arguments:
        A {numpy.ndarray} -- matrix whose columns form basis for subspace U
        B {numpy.ndarray} -- matrix whose columns form baiss for subspace W
        v0 {numpy.ndarray} -- initialization vector
        n {int} -- number of sweeps for alternating projection
    Returns:
        v {numpy.ndarray} -- the output after 2n steps of alternating projection
        err {numpy.ndarray} -- the error after each full pass
    """
    uw = np.hstack([A, B]) @ null_space(np.hstack([A, -B]))
    if uw.shape[1] == 0:
        v0_uw = np.zeros(v0.shape[0])
    else:      
        puw = np.linalg.lstsq(uw,v0,rcond=None)[0]
        v0_uw = np.dot(uw,puw)
    v = v0
    err = []
    
    for i in range(2*n):
        if i%2==0:
            pa = np.linalg.lstsq(A,v,rcond=None)[0]
            v = np.dot(A,pa)
        else:
            pb = np.linalg.lstsq(B,v,rcond=None)[0]
            v = np.dot(B,pb)
        
            err_v = np.abs(v-v0_uw)
            err.append(np.max(err_v))
    
    err = np.array(err)
    
    return v, err

# Exercise 2: Kaczmarz algorithm for solving linear systems
def kaczmarz(A, b, I):
    """
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the Kaczmarz algorithm
    Returns:
        X {numpy.ndarray} -- the output of all I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here
    m = A.shape[0]
    v = np.zeros(A.shape[1])
    X = []
    err = []
    
    for i in range (I):
        for j in range(m):
            pj = (np.dot(v,A[j])-b[j])/(np.dot(A[j],A[j]))
            v = v - pj*A[j]
            
        X.append(v)
        err_v = np.abs(np.dot(A,v)-b)
        err.append(np.max(err_v))
    
    X = np.array(X)
    X = X.T
    err = np.array(err)

    return X, err

# Exercise 4: Alternating projection to satisfy linear inequalities
def lp_altproj(A, b, I, s=1):
    """
    Find a feasible solution for A v >= b using alternating projection
    starting from v0 = 0
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the alternating projection
        s {numpy.float} -- step size of projection (defaults to 1)
    Returns:
        v {numpy.ndarray} -- the output after I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    # Add code here
    m = A.shape[0]
    v = np.zeros(A.shape[1])
    err = []
    
    for i in range(I):
        for j in range(m):
            if np.dot(v,A[j])>=b[j]:
                v = v
            else:
                pj = (np.dot(v,A[j])-b[j])/(np.dot(A[j],A[j]))
                v = v - s*pj*A[j]
        
        err_v = b-np.dot(A,v)
        err.append(np.max(err_v))
        
    err = np.array(err)
    
    return v, err
