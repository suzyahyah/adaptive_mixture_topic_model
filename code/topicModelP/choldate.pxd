
cimport numpy as np
ctypedef np.float64_t FLOAT_t

cpdef np.ndarray cholupdate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x)

cpdef np.ndarray choldowndate(np.ndarray[FLOAT_t, ndim=2] R, np.ndarray[FLOAT_t, ndim=1] x)
