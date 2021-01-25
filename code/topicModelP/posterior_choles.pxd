cimport numpy as np
ctypedef np.float64_t DTYPE_t

cdef class Posterior:
    cdef public int ntopics, ndim, nu0, dof0, dof0_var
    cdef public double kln2pi, mu0_b
    cdef public set update_logdet_k

    cdef double[:] cov_logdet_allk, scaleTdist_allk, betab_allk
    cdef np.ndarray upperT_allk, mu0, sum_embed_allk, mu_allk, sigma0, upperT_0, scaled_cov_allk, sigma_allk, cov_allk, cov_inv_allk
    
  
    cpdef void init_full(self, int[:] all_topiccounts, sum_embed_allk, scaled_cov_allk) except *

    cpdef void init_partial(self) except *

    cpdef update(self, Py_ssize_t k, np.ndarray[DTYPE_t, ndim=1] word_emb, int nwords_k, str add_drop)
    cpdef calc_mv_ndens(self, k, word_emb)

    cpdef double calc_mv_tdens(self, Py_ssize_t k, int nwords_k, np.ndarray[DTYPE_t, ndim=1] word_emb) except*

    cpdef void load(self, str pickle_dir, str suffix=*) except *

    cpdef void save(self, str pickle_dir, str suffix=*) except *
    cpdef np.ndarray get_mu_allk(self)
    cpdef np.ndarray get_sum_embed_allk(self)
    cpdef np.ndarray get_upperT_allk(self)
    cpdef double[:] get_betab_allk(self)
    cpdef double get_covlogdet_allk(self, int k)

    cpdef void set_dof0_var(self, int dof0)
