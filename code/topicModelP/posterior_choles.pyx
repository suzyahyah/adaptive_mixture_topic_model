###!/usr/bin/python
# Author: Suzanna Sia
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration


#### cython c div =True doesnt work for some reason
import os
import pickle
import numpy as np
np.random.seed(1)
cimport numpy as np
np.random.seed(1)
import scipy as sp

from scipy.stats import multivariate_normal as mvn
import pdb
import math
#import cholesky
cimport choldate
from libc.math cimport lgamma, pow, pi, log,sqrt
from sklearn.covariance import LedoitWolf

DTYPE = np.float64
#ctypedef np.float64_t DTYPE_t

cdef class Posterior():

#    cdef public int ntopics, ndim, nu0, dof0
#    cdef public double kln2pi
#    cdef public double[:] mu0, cov_logdet_allk
#    cdef public double[:, :] sum_embed_allk, mu_allk, sigma0, upperT_0
#    cdef public double[:, :, :] scaled_cov_allk, sigma_allk, cov_allk, cov_inv_allk #, upperT_allk
#    cdef np.ndarray upperT_allk

#    cdef public double mu0_b


    def __init__(self, ntopics, ndim, mu0=None, mu0_b=1, sigma0=None):
            
        self.ntopics = ntopics
        self.mu0 = mu0
        self.mu0_b = mu0_b #0.1 # same as gaussianLDA setting
        self.sigma0 = sigma0
        self.ndim = ndim
        self.dof0 = ndim+2 #nu0
        self.dof0_var = ndim+2 #nu0

        self.sum_embed_allk = np.zeros((self.ntopics, self.ndim), dtype=DTYPE)
        self.scaled_cov_allk = np.zeros((self.ntopics, self.ndim, self.ndim), dtype=DTYPE)

        self.mu_allk = np.zeros((self.ntopics, self.ndim), dtype=DTYPE)
        self.sigma_allk = np.zeros((self.ntopics, self.ndim, self.ndim), dtype=DTYPE)
        self.cov_allk = np.zeros((self.ntopics, self.ndim, self.ndim), dtype=DTYPE)
        self.cov_inv_allk = np.zeros((self.ntopics, self.ndim, self.ndim), dtype=DTYPE)
        self.cov_logdet_allk = np.zeros(self.ntopics, dtype=DTYPE)
        self.scaleTdist_allk = np.zeros(self.ntopics, dtype=DTYPE)
        self.upperT_allk = np.zeros((self.ntopics, self.ndim, self.ndim), dtype=DTYPE)
        self.betab_allk = np.zeros(self.ntopics, dtype=DTYPE)
        self.upperT_0 = sp.linalg.cholesky(sigma0, lower=False)

        self.kln2pi = self.ndim * np.log(2 * np.pi)
        self.update_logdet_k = set(range(self.ntopics))

        
    cpdef void init_partial(self) except *:
        for k in range(self.ntopics):
            self.mu_allk[k] = np.array(self.mu0, copy=True)
            self.upperT_allk[k] = np.array(self.upperT_0, copy=True)
            self.betab_allk[k] \
            = (self.ndim/(np.sum(np.asarray(self.upperT_allk[k]).diagonal())))/self.ntopics
            #= self.ndim/(np.sum(np.asarray(self.upperT_allk[k]).diagonal()**2)+300)

    cpdef void init_full(self, int[:] all_topiccounts, sum_embed_allk, scaled_cov_allk) except * :

        cdef double mu_k_b, norm, dof_k, cov_logdet_k
        cdef int nwords_k
        cdef Py_ssize_t k
        cdef np.ndarray[DTYPE_t, ndim=1] sum_embed_k, A, mu_k
        cdef double[:, :] iden, sigma_k, cov_k, upperT, cov_inv_k
        cdef np.ndarray[DTYPE_t, ndim=2] Ak 

        self.sum_embed_allk = sum_embed_allk
        self.scaled_cov_allk = scaled_cov_allk

        for k in range(self.ntopics):
            
            nwords_k = all_topiccounts[k]
            sum_embed_k = np.asarray(self.sum_embed_allk[k])

            # mu_k update

            kk = self.mu0_b + nwords_k
            nu_k = self.dof0 + nwords_k
            mu_k = (self.mu0_b * np.asarray(self.mu0) + sum_embed_k)/kk
            
            self.mu_allk[k] = mu_k

            # sigma_k update
            norm = (self.mu0_b * nwords_k) /kk 
            A = sum_embed_k/nwords_k - self.mu0
            Ak = np.outer(A, A)
            #Ak = np.matmul(A, A.T)
            sigma_k = self.sigma0 + np.asarray(self.scaled_cov_allk[k]) + norm * Ak
            self.sigma_allk[k] = sigma_k

               # ===========
                # cholesky
            upperT = sp.linalg.cholesky(sigma_k, lower=False)
            assert np.isnan(upperT).sum()==0, "upperT contains nan"
            assert np.isinf(upperT).sum()==0, "upperT contains inf"
            self.upperT_allk[k] = upperT

            # will be updated by self.update_log_det
            self.betab_allk[k] \
            = (self.ndim/(np.sum(np.asarray(self.upperT_allk[k]).diagonal())))/self.ntopics
            #= self.ndim/sqrt(np.sum(np.asarray(self.upperT_allk[k]).diagonal()**2))
                #= self.ndim/sqrt(np.sum(np.asarray(self.upperT_allk[k]).diagonal()))
               

    #cpdef void update(self, Py_ssize_t k, np.ndarray[DTYPE_t, ndim=1] word_emb, int nwords_k, str add_drop) except *:
    cpdef update(self, Py_ssize_t k, np.ndarray[DTYPE_t, ndim=1] word_emb, int nwords_k, str add_drop):
        cdef np.ndarray[DTYPE_t, ndim=1] x
        cdef double kk, nu_k

        kk = self.mu0_b + nwords_k
        nu_k = self.dof0 + nwords_k
        self.update_logdet_k.add(k)


        if add_drop == "add":
            # update mean before adding
            self.mu_allk[k] = ((kk-1)*self.mu_allk[k] + word_emb)/kk
            x = self.mu_allk[k] - word_emb
            
            # CHOLESKY 
            x = sqrt(kk/(kk-1)) * x
            self.upperT_allk[k] = choldate.cholupdate(self.upperT_allk[k], x)
   
        if add_drop == "drop":
            # update mean after dropping
            x = self.mu_allk[k] - word_emb

            x *= sqrt((kk+1)/kk)
            self.upperT_allk[k] = choldate.choldowndate(self.upperT_allk[k], x)
            self.mu_allk[k] = ((kk+1)*self.mu_allk[k] - word_emb)/kk



    cpdef calc_mv_ndens(self, k, word_emb):
        return mvn.logpdf(word_emb, np.asarray(self.mu_allk[k]), cov=np.identity(self.ndim))

    cpdef double calc_mv_tdens(self, Py_ssize_t k, int nwords_k, np.ndarray[DTYPE_t, ndim=1] embed_dist) except *:
        # Equation from Wikipedia:
        # Implementation from: https://github.com/scikit-learn/scikit-learn/blob/20661b5018ba45f4452218a689640ff007cd361f/sklearn/mixture/gmm.py#L720
        #

        cdef np.ndarray[DTYPE_t, ndim=1] Lb
        cdef np.ndarray[DTYPE_t, ndim=2] cov_k
        cdef np.ndarray[DTYPE_t, ndim=2] upperT
        cdef double log_llh, logdet_cov, middle, nu_n, scaleTdist, nu, norm, inner
        cdef double[:,:] tmp
        cdef double[:] mu_k
        cdef np.intp_t nd, i, j
        
        
#        nu_n = nwords_k + self.dof0 # when we make self.dof0 negative
#        kk = self.mu0_b + nwords_k
#        scaleTdist = sqrt((kk+1)/(kk*(nu_n - self.ndim +1)))
        nu = self.dof0_var + nwords_k - self.ndim + 1

        # if the fella has been updated in add or drop operation
        if k in self.update_logdet_k:
            # self.dof0 should be self.ndim + 2, it doesnt change
            nu_n = nwords_k + self.dof0 # when we make self.dof0 negative
            kk = self.mu0_b + nwords_k
            scaleTdist = sqrt((kk+1)/(kk*(nu_n - self.ndim +1)))
            #scaleTdist = sqrt((kk+1)/(kk*(n_words_k - self.ndim +1)))
            # now this is true L
            upperT = self.upperT_allk[k] * scaleTdist

            logdet_cov = 2*np.sum(np.log(np.diagonal(upperT)))
            self.cov_logdet_allk[k] = logdet_cov
            self.scaleTdist_allk[k] = scaleTdist

            # gao dim
            #self.betab_allk[k] = self.ndim/sqrt(np.sum(np.asarray(self.upperT_allk[k]).diagonal()**2))
            # this upperT has been scaled
            self.betab_allk[k] = (self.ndim/(np.sum(np.asarray(upperT).diagonal())))/self.ntopics
            self.update_logdet_k.remove(k)

        else:
            logdet_cov = self.cov_logdet_allk[k]
            scaleTdist = self.scaleTdist_allk[k]
            upperT = self.upperT_allk[k] * scaleTdist

        Lb = sp.linalg.solve_triangular(upperT, embed_dist, lower=False, check_finite=False) 
        middle = np.inner(Lb, Lb)

        #log_llh = -0.5 * (logdet_cov + middle + self.kln2pi)

        norm = (lgamma((nu + self.ndim) / 2.) \
                - lgamma(nu / 2.) \
                - 0.5 * self.ndim * (log(nu * pi)))

        inner = - ((nu + self.ndim) / 2) * log(1+(middle / nu))
        log_llh = norm + inner - logdet_cov

        return log_llh


    cpdef void load(self, str pickle_dir, str suffix="") except *:
        load_name = os.path.join(pickle_dir, self.__class__.__name__+suffix)

        load1 = load_name+"_sum_embed_allk"
        self.sum_embed_allk = pickle.load(open(load1, 'rb'))

        load2 = load_name+"_mu_allk"
        self.mu_allk = pickle.load(open(load2, 'rb'))

        load3 = load_name + "_upperT_allk"
        self.upperT_allk = pickle.load(open(load3, 'rb'))

        for k in range(self.ntopics):
            #self.betab_allk[k] = self.ndim/sqrt(np.sum(np.asarray(self.upperT_allk[k]).diagonal()**2))
            # will be updated later by self.update_logdet
            self.betab_allk[k] = (self.ndim/(np.sum(np.asarray(self.upperT_allk[k]).diagonal())))/self.ntopics
            

#        for k in range(self.ntopics):
#            assert not np.isnan(np.asarray(self.upperT_allk[k])).any(), (k)
#            assert not np.isnan(np.asarray(self.mu_allk[k])).any(), (k)
        #for k in range(self.ntopics):
        #    self.upperT_allk[k] += np.identity(self.ndim)*0.01

        

    cpdef void save(self, str pickle_dir, str suffix="") except *:
        save_name = os.path.join(pickle_dir, self.__class__.__name__+suffix)

        save1 = save_name+"_sum_embed_allk"
        #pickle.dump(self.sum_embed_allk.base, open(save1, 'wb'))
        pickle.dump(self.sum_embed_allk, open(save1, 'wb'))

        save2 = save_name+"_mu_allk"
        #pickle.dump(self.mu_allk.base, open(save2, 'wb'))
        pickle.dump(self.mu_allk, open(save2, 'wb'))

        save3 = save_name+"_upperT_allk"
        pickle.dump(self.upperT_allk, open(save3, 'wb'))

    cpdef np.ndarray get_mu_allk(self):
        return self.mu_allk

    cpdef np.ndarray get_upperT_allk(self):
        return self.upperT_allk

    cpdef np.ndarray get_sum_embed_allk(self):
        return self.sum_embed_allk

    cpdef double[:] get_betab_allk(self):
        return self.betab_allk
#        return self.cov_logdet_allk

    cpdef double get_covlogdet_allk(self, int k):
        return self.cov_logdet_allk[k]
#

    cpdef void set_dof0_var(self, int dof0):
        self.dof0_var = dof0



