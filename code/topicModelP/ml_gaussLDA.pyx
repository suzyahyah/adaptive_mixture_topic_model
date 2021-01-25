#!usr/bin/python
# Author: Suzanna Sia
#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
np.random.seed(1)
cimport numpy as np
np.random.seed(1)

import random
random.seed(1)

import math
import pickle
import pstats, cProfile
import os, sys, time
import gensim
from scipy.special import loggamma
from scipy.stats import multivariate_normal
import scipy as sp


from posterior_choles cimport Posterior
from report_utils import print_top_words, calc_hgs, topic_coherence, topic_coherence2
import generic_utils

from sklearn.cluster import KMeans
from libc.math cimport exp, log, sqrt
from libc.stdlib cimport rand, RAND_MAX, srand
srand(1)

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef class ml_gaussLDA():

    cdef int ntopics, niter, ndim, nl, shared_params, vocab_size, \
    prior_mean, stagger, save_every

    cdef double alpha_sum, alpha_k, alpha_2k, test_coh, train_coh, intp_prior
    cdef double beta, MAXENT, interpolate, mu0_b, cov0_scalar, l_beta, beta_beta

    cdef dict l_posterior, oov_word_topic 
    cdef double[:] l_beta_sum, l_intp_prior
    cdef int[:, :] l_doc_lengths
    cdef int[:, :] l_alltopic_counts
    cdef int[:, :, :] l_word_topic_counts, l_d_k, l_topicAssign_doc_word, l_doc_vecs
    cdef int MAX_DOC_LEN, MAX_VOCAB_SIZE, max_eval, scaling_dof0, seed

    cdef str pickle_dir, train_prop_dir, llh_dir, hgs_dir
    cdef list topic_assignment, langs


    def __init__(self, languages, ntopics, ndim, niter, alpha_sum, beta, 
            pickle_dir, train_prop_dir, llh_dir, hgs_dir, topic_assignment=[], 
            interpolate=1, shared_params=0, prior_mean=0, stagger=0, mu0_b=0, 
            cov0_scalar=1, max_eval=50, scaling_dof0=0, beta_beta=10, seed=0):

        # multi-lingual
        self.langs = languages
        self.seed = seed
        self.nl = len(languages)
        self.vocab_size = generic_utils.get_max_vocab_size(self.langs)
           
        # training parameters
        self.ntopics = ntopics
        self.niter = niter
        self.test_coh = 0
        self.train_coh = 0

        # hyperparameters
        #self.alpha_sum = alpha_sum
        #self.alpha_k = np.array([alpha_sum/ntopics for i in range(ntopics)], dtype=DTYPE)[0]
        #self.alpha_k = np.array([0.01*ntopics for i in range(ntopics)], dtype=DTYPE)[0]
        # new alpha_sum
        self.alpha_k = 0.01 * ntopics
        self.alpha_sum = self.alpha_k * ntopics
        self.alpha_2k = self.alpha_k * 2

        self.beta = beta
        self.l_beta = 0.0
        self.l_beta_sum = np.array([beta * self.langs[l].vocab_size for l in range(self.nl)], dtype=DTYPE)
        self.beta_beta = beta_beta # beta parameter on beta distribution
        self.mu0_b = mu0_b
        self.cov0_scalar = cov0_scalar
        
        # counts
        self.MAX_DOC_LEN = max([self.langs[el].max_length for el in range(self.nl)]) #var no.
        self.MAX_VOCAB_SIZE = max([self.langs[el].vocab_size for el in range(self.nl)]) #var no.
        self.l_intp_prior = np.zeros(self.nl, dtype=DTYPE) #, self.langs[el].vocab_size/self.langs[0].ndocs
        # The larger the vocab size, the less reliable the estimates are.
        # The smaller the vocab size, the more reliable the estimates are.

        self.l_doc_vecs = np.zeros((self.nl, self.langs[0].ndocs, self.MAX_DOC_LEN), dtype=np.intc)
        self.l_topicAssign_doc_word = np.zeros((self.nl, self.langs[0].ndocs, self.MAX_DOC_LEN), dtype=np.intc)
        self.l_word_topic_counts = np.zeros((self.nl, self.vocab_size, ntopics), dtype=np.intc)
        self.l_alltopic_counts = np.zeros((self.nl, ntopics), dtype=np.intc)
        self.l_d_k = np.zeros((self.nl, self.langs[0].ndocs, ntopics), dtype=np.intc)
        self.l_doc_lengths = np.zeros((self.nl, self.langs[0].ndocs), dtype=np.intc)

        # Embeddings
        self.ndim = ndim  
        self.l_posterior = {}

        #control-logic
        self.topic_assignment = topic_assignment
        self.shared_params = int(shared_params)

        if self.shared_params == 100:
            self.save_every = 10
        else:
            self.save_every = 5

        self.MAXENT = sp.stats.entropy(np.asarray([1/ntopics for i in range(ntopics)]))
        self.prior_mean = prior_mean
        self.interpolate = interpolate
        self.stagger = stagger
        self.max_eval = max_eval
        self.scaling_dof0 = scaling_dof0

        self.pickle_dir = pickle_dir
        self.train_prop_dir = train_prop_dir
        self.llh_dir = llh_dir
        self.hgs_dir = hgs_dir

        self.oov_word_topic = {}


    cpdef void initialise(self) except *:

        cdef Py_ssize_t l, k 
        cdef object lang
        cdef int[:] temp

        for l in range(self.nl):
            temp = np.asarray(self.langs[l].get_doc_lengths(), dtype=np.intc)
            self.l_doc_lengths[l] = temp
            
            # this doesnt work
            #self.l_doc_lengths[l] = np.asarray(self.langs[l].get_doc_lengths())
            lang = self.langs[l]
            self.assign_topics(l, lang, lang.train_data, mode="train")
            print(f"assigned topics for {lang.name}")
            self.l_intp_prior[l] = self.langs[l].vocab_size/self.langs[0].ndocs
            
        self.calc_params()

        for l in range(self.nl):
            counts_arr = np.asarray(self.l_alltopic_counts[l])
            print(f"{lang.name} topic counts:", np.min(counts_arr), "--", np.max(counts_arr))

        print("Completed initialisation")

    cpdef calc_beta_values(self, fn):

        for l in range(self.nl):
            expected_alpha = 0
            j=0
            for k in range(self.ntopics):
                for w in range(self.langs[l].vocab_size):
                    alpha = self.langs[l].vocab_prior[w] + self.l_word_topic_counts[l, w,k] + 1/self.l_intp_prior[l]
                    expected_alpha += alpha
                    j+=1

            expected_alpha = expected_alpha/j
        expected_beta = 0

        for k in range(self.ntopics):
            expected_beta += (self.l_posterior[0].get_betab_allk()[k]+0.01)
        expected_beta = expected_beta/self.ntopics
        
        total_words = np.sum(self.l_alltopic_counts[l])
        towrite = "{}\t{}\t{}\t{}\n".format(self.ntopics,
                np.around(self.l_intp_prior[l], 3),
            np.around(expected_alpha, 3), np.around(expected_beta, 3))

        print("write to:", fn)
        with open(fn, 'a') as f:
            f.write(towrite)


    cdef get_mean(self, l=0):
        if not self.prior_mean==1:
            mean = np.zeros(self.ndim)
        elif self.shared_params==1 and self.nl>1:
            mean = (self.langs[0].get_sample_mean()*self.langs[0].vocab_size \
                    + self.langs[1].get_sample_mean()*self.langs[1].vocab_size)/ \
                    (self.langs[0].vocab_size + self.langs[1].vocab_size)
        else:
            mean = self.langs[l].get_sample_mean()
        return mean

    cdef get_cov0(self, l=0):
        if self.cov0_scalar>0:
            sample_cov = np.identity(self.ndim)*self.cov0_scalar

        elif self.shared_params==1 and self.nl>1:
            print("using pooled sample covariance")
            # pooled sample_cov
            sample_cov1 = self.langs[0].get_sample_cov()
            sample_cov2 = self.langs[1].get_sample_cov()
            sample_cov = (self.langs[0].vocab_size*sample_cov1
            + self.langs[1].vocab_size*sample_cov2)/(self.langs[0].vocab_size+self.langs[1].vocab_size-2)
        else:
            sample_cov = self.langs[l].get_sample_cov()
        return sample_cov



    cpdef calc_params(self):
        cdef Py_ssize_t ix, i
        cdef int[:] doc_vec

        print("calculating params..", "nl:", self.nl, "sharedparams:", self.shared_params)
        # given assigned l_doc_vecs, l_topicAssign_doc_word
        # calculate params for shared_params1 and shared_params0
       
        l_sum_embed_allk = {}

        if self.nl==1:
            sample_mean = self.get_mean(0)
            sample_cov = self.get_cov0(0)
            topic_counts = self.l_alltopic_counts[0]
            ## Initialise Posterior object ##
            self.l_posterior[0] = Posterior(self.ntopics, self.ndim, mu0=sample_mean,\
                    sigma0=sample_cov, mu0_b=self.mu0_b)

            sum_embed_allk = self.calc_par_allk(process_langs=range(self.nl), mode="sum_embed")
            topic_means_allk = self.calc_topic_means_allk(sum_embed_allk, topic_counts)
            scaled_cov_allk = self.calc_par_allk(topic_means_allk,
                    process_langs=range(self.nl), mode="cov")
            self.l_posterior[0].init_full(topic_counts, sum_embed_allk, scaled_cov_allk)
        else:
            if self.shared_params == 1:
                
                sample_mean = self.get_mean(0)
                sample_cov = self.get_cov0(0)
     
                self.l_posterior[0] = Posterior(self.ntopics, self.ndim, mu0=sample_mean, \
                                        sigma0=sample_cov, mu0_b=self.mu0_b)

                alltopic_counts = np.asarray(self.l_alltopic_counts[0]) + np.asarray(self.l_alltopic_counts[1])
                sum_embed_allk = self.calc_par_allk(process_langs=range(self.nl), mode="sum_embed")
                topic_means_allk = self.calc_topic_means_allk(sum_embed_allk, alltopic_counts)
                scaled_cov_allk = self.calc_par_allk(topic_means_allk, process_langs=range(self.nl), mode="cov")
                self.l_posterior[0].init_full(alltopic_counts, sum_embed_allk, scaled_cov_allk)
            else:
                print("shared params, no param init")

   
    cpdef calc_topic_means_allk(self, sum_embed_allk, alltopic_counts):

        topic_means_allk = np.zeros((self.ntopics, self.ndim))
        for k in range(self.ntopics):
            topic_means_allk[k] = sum_embed_allk[k] / alltopic_counts[k]
        return topic_means_allk


    cpdef calc_par_allk(self, topic_means_allk=None, process_langs=[], mode=""):
        # Calculates parameters for either sum_embed_allk or cov_allk

        if mode=="cov":
            # scaled_cov_allk
            par_allk = np.zeros((self.ntopics, self.ndim, self.ndim), dtype=DTYPE)
        elif mode=="sum_embed":
            # sum_embed_allk
            par_allk = np.zeros((self.ntopics, self.ndim))

        for l in process_langs:
            lang = self.langs[l]
            for d in range(len(lang.train_data)):
                doc_vec = self.l_doc_vecs[l, d]
                topic_sequence = self.l_topicAssign_doc_word[l, d]
                doc_len = self.l_doc_lengths[l, d]

                for i in range(doc_len):
                    ix = doc_vec[i]
                    k = topic_sequence[i]
                    word_emb = lang.get_emb2(ix)

                    if mode=="cov":
                        mean_centered = word_emb - topic_means_allk[k]
                        par_allk[k] += np.outer(mean_centered, mean_centered)

                    elif mode=="sum_embed":
                        par_allk[k] += word_emb

        return par_allk 
 

    def infer_test_distributions(self, test_data, iters=10, l=0):
        test_data, _ = self.langs[l].prep_data(test_data)
        test_data = self.langs[l].filter_oov_embeddings(test_data, test_mode=True)

        lang = self.langs[l]

        self.update_corpus_index(test_data)
        self.assign_topics(l, lang, test_data, mode="test")
        
        total_n = self.langs[l].ndocs + len(test_data)
        other_l = 1 - l

        for itr in range(iters):
            print("test itr:", itr)
            for d in range(self.langs[l].ndocs, total_n):
                doc_vec = self.l_doc_vecs[l, d]
                doc_len = self.l_doc_lengths[l, d]

                self.l_beta = self.l_beta_sum[l]
                self.sample_one_doc(l, other_l, d, doc_vec, doc_len)

        topicAssign_doc_word = []
        doc_lengths = []
        doc_vecs = []

        # this is the testing range
        for d in range(self.langs[l].ndocs, total_n):
            # currently takes most likely topic for that word. but what if we do proportions?
            topicAssign_doc_word.append(self.l_topicAssign_doc_word[l, d])
            doc_lengths.append(self.l_doc_lengths[l, d])
            doc_vecs.append(self.l_doc_vecs[l, d])


        if self.shared_params!=100:
            for d in range(len(topicAssign_doc_word)):
                doc_position_topic = topicAssign_doc_word[d]
                doc_len = doc_lengths[d]
                doc_vec = doc_vecs[d]

                for w in range(doc_len):
                    topic = doc_position_topic[w]
                    if topic==-99:
                        oov_ix = doc_vec[w]
                        word = lang.oov_ix2w[oov_ix]

                        #print("l:", l)
                        #print("word:", word)

                        topic = self.predict_oov_word_topic(l, word) 

                    doc_position_topic[w] = topic

                topicAssign_doc_word[d] = doc_position_topic

        all_topic_probs, invalid_doc_ix = generic_utils.calc_topic_proportions(self.ntopics,
                topicAssign_doc_word, doc_lengths)



        return all_topic_probs, invalid_doc_ix

    cdef update_corpus_index(self, data):
        if self.l_d_k.shape[1]==self.langs[0].ndocs:
            test_l_d_k = np.zeros((self.nl, len(data), self.ntopics), dtype=np.intc)
            self.l_d_k = np.append(self.l_d_k, test_l_d_k, axis=1)

            test_doc_vecs = np.zeros((self.nl, len(data), self.MAX_DOC_LEN), dtype=np.intc)
            # something weird with l_doc_vecs for 1 0 0 
            print(test_doc_vecs.shape, self.l_doc_vecs.shape)
            self.l_doc_vecs = np.append(self.l_doc_vecs, test_doc_vecs, axis=1)

            test_doc_position_topic = np.zeros((self.nl, len(data), self.MAX_DOC_LEN), dtype=np.intc)
            self.l_topicAssign_doc_word = np.append(self.l_topicAssign_doc_word, test_doc_position_topic, axis=1)

            test_doc_lengths = np.zeros((self.nl, len(data)), dtype=np.intc)
            self.l_doc_lengths = np.append(self.l_doc_lengths, test_doc_lengths, axis=1)


    cdef assign_topics(self, int l, object lang, list data, str mode=""):
        """ Assigns topics to words in doc"""

        cdef int w, d, topic, ix
        cdef int[:] doc_vec
        cdef int[:] doc_position_topic

        for d in range(len(data)):

            words = data[d]
            doc_vec = np.zeros(self.MAX_DOC_LEN, dtype=np.intc)
            doc_position_topic = np.zeros(self.MAX_DOC_LEN, dtype=np.intc)
            
            if mode == "test":
                d = self.langs[0].ndocs + d

            len_words=0
            for w in range(len(words)):
                
                if w>(self.MAX_DOC_LEN-1):
                    break

                word = words[w]
                len_words +=1

                if mode=="test":
                    if word in lang.w2ix:
                        ix = lang.w2ix[word]
                        topic = np.argmax(self.l_word_topic_counts[l, ix])
                    else:
                        doc_position_topic[w] = -99
                        doc_vec[w] = lang.oov_w2ix[word]
                        continue

                if mode=="train":
                    ix = lang.w2ix[word]
                    topic = random.randint(0, self.ntopics-1)
               
                doc_position_topic[w] = topic
                doc_vec[w] = ix
                
                self.l_word_topic_counts[l, ix, topic] += 1
                self.l_alltopic_counts[l, topic] += 1
                self.l_d_k[l, d, topic] += 1

                # For use in posterior_choles.Posterior
                word_emb = np.asarray(lang.get_emb(word), dtype=DTYPE)
                
                nwords_k = self.l_alltopic_counts[l, topic]

                if mode=="test":
                    if self.shared_params==1:
                        self.l_posterior[0].update(topic, word_emb, nwords_k, "add")
                    elif self.shared_params==0 or self.shared_params==99:
                        self.l_posterior[l].update(topic, word_emb, nwords_k, "add")
            
            #if len(doc_vec)>self.MAX_DOC_LEN:
            #    print("doc len larger")
            #    doc_vec=doc_vec[:self.MAX_DOC_LEN]
            #print(len(doc_vec), self.l_doc_vecs.shape, self.MAX_DOC_LEN)

            self.l_doc_vecs[l, d] = doc_vec
            #print('a')
            self.l_topicAssign_doc_word[l, d] = doc_position_topic
            #print('b')
            self.l_doc_lengths[l, d] = min(self.MAX_DOC_LEN, len_words)
            #print('c')

    cpdef double get_test_coh(self):
        return self.test_coh
    cpdef double get_train_coh(self):
        return self.train_coh

    cpdef void run_gibbs_sampling(self, int niter, int loaditer=0, te_lang=None) except *:
        self._run_gibbs_sampling(niter, loaditer, te_lang)

    cdef void _run_gibbs_sampling(self, int niter, int loaditer=0, object te_lang=None) except *:
        cdef str loop
        cdef double llh
        cdef Py_ssize_t d, l, other_lang, itr, cycle
        cdef int doc_len
        cdef int[:] doc_vec
        cdef int[:,:] doc_vecs

        niter += loaditer

        if self.shared_params==101:
            cycle=1
        else:
            cycle=0
        
        print(loaditer, niter)
        for itr in range(loaditer, niter):
            start = time.time()
            print("--iteration:", itr)
            sys.stdout.flush()

            #if itr%5==0:
            #betab_allk = self.l_posterior[0].get_betab_allk()
           
            #l_topic_scores = topic_coherence(self.nl, self.langs, self.l_word_topic_counts,\
            #            25, self.ntopics)
            #print_top_words(self.nl, self.langs, self.l_word_topic_counts, 20,\
            #            self.ntopics, betab_allk, l_topic_scores=l_topic_scores)



            ##############################
            ### Experiment 1: Stagger CL-LDA and CL-ML-LDA
            if self.stagger>0:
                if itr<=self.stagger:
                    print("Stagger: Running CL vanilla LDA")
                    self.shared_params = 100
                else:
                    print("Stagger: Running CL gaussian LDA (shared1)")
                    self.shared_params = 1
                    if itr==(self.stagger+1):
                        self.calc_params()
            ### ##########################
            ### Experiment 2: Cyclic training
            if cycle==1:
                if itr%2==0:
                    self.shared_params=1
                else:
                    self.shared_params=101
            ##############################
            # MAIN
            
            for d in range(len(self.l_doc_vecs[0])):
                # shuffled indexes
                if d%1000==0 and d>0:
                    print("document:", d)
                    sys.stdout.flush()
                    self.print_entropy_stats()

                for l in range(self.nl):
                    other_lang = 1 - l # 0 or 1
                    doc_vec = self.l_doc_vecs[l][d]
                    doc_len = self.l_doc_lengths[l, d]

                    self.l_beta = self.l_beta_sum[l]
                    self.sample_one_doc(l, other_lang, d, doc_vec, doc_len)
                
            #if itr%self.save_every==0:
                #suffix="niter{}".format(str(itr))
                #self.save(self.pickle_dir, suffix=suffix)

            if itr%5==0:
                if self.shared_params != 100:
                    betab_allk = self.l_posterior[0].get_betab_allk()
           #     l_topic_scores = topic_coherence(self.nl, self.langs, self.l_word_topic_counts,\
           #             30, self.ntopics)
                if len(te_lang) == 0:
                    continue
                else:
                    test_lts  = topic_coherence2(self.nl, te_lang, self.l_word_topic_counts,
                        10, self.ntopics, te_lang[0].ndocs, mode="npmi")

                    self.test_coh = (np.round(np.mean(test_lts), 5))
                    print("test coherence:", self.test_coh)

                    train_lts= topic_coherence2(self.nl, self.langs, self.l_word_topic_counts,
                        10, self.ntopics, self.langs[0].ndocs, mode="npmi")

                self.train_coh = np.round(np.mean(train_lts),5)
                print("train coherence:", self.train_coh)


                if self.shared_params != 100:
                    print_top_words(self.nl, self.langs, self.l_word_topic_counts, 20,\
                            self.ntopics, betab_allk, l_topic_scores=train_lts)

                suffix="niter{}.n{}.s{}".format(str(itr), str(self.langs[0].ndocs), str(self.seed))
                self.save(self.pickle_dir, suffix=suffix)



            ################################
            # SIMULATION
            #
            if self.langs[0].name=="a":
                hgs_scores = calc_hgs(self.langs, self.l_word_topic_counts)
                for l in range(self.nl):
                    hgs_score = hgs_scores[l]
                    if self.interpolate==0:
                        intp = "0"
                    else:
                        intp = str(self.interpolate)

                    fn = "{}-{}-{}-{}".format(self.shared_params, intp, \
                            self.stagger, self.langs[l].name)
                    print("saving to:", self.hgs_dir, fn)
                    hgs_outf = os.path.join(self.hgs_dir, fn)
                    with open(hgs_outf, 'a') as f:
                        f.write("\n{}\t{}".format(itr, hgs_score))


            print("time;", time.time()-start)

        suffix="niter{}.n{}.s{}".format(str(itr), str(self.langs[0].ndocs), str(self.seed))
        self.save(self.pickle_dir, suffix=suffix)


    def write_top_words(self, wordsout_dir):
        print_top_words(self.nl, self.langs, self.l_word_topic_counts, 20, self.ntopics,
                self.l_posterior[0].get_upperT_allk(), savepath=wordsout_dir)

    def print_entropy_stats(self, train_prop_dir=""):
        for l in range(self.nl):
            topic_counts = np.asarray([np.asarray(self.l_word_topic_counts[l,:, k]).sum() for k in range(self.ntopics)])
            topic_props = topic_counts/topic_counts.sum()
            topic_props = np.around(topic_props, decimals=3)
            entropy = sp.stats.entropy(topic_props)

        #    print("l:{} topics:{}".format(l, topic_props))
            print("\n{} max topic {}:{:.3f}, min:{:.3f}, nzeros:{}".format( \
                self.langs[l].name, \
                np.argmax(topic_props),\
                topic_props[np.argmax(topic_props)], \
                topic_props[np.argmin(topic_props)], \
                self.ntopics-np.count_nonzero(topic_counts/topic_counts.sum())))

            with open(train_prop_dir+"_ent_{}".format(self.langs[l].name), 'a') as f:
                f.write("{:.3f}\n".format(entropy/self.MAXENT))

            print("Entropy(mixing ratio):{:.5f} Interpolate:{}".format(entropy/self.MAXENT, self.interpolate))

            
            if len(train_prop_dir)>0:
                np.savetxt(train_prop_dir+"_{}".format(self.langs[l].name), topic_props, fmt='%.3f')



    cdef int get_nwords(self, int l, Py_ssize_t k) except *:
        cdef int nwords_k = 0

        if self.shared_params==1 and self.nl>1:
            nwords_k = self.l_alltopic_counts[0,k] + self.l_alltopic_counts[1,k]
        else:
            nwords_k = self.l_alltopic_counts[l, k]

        return nwords_k

    cdef double get_sum_d(self, other_l, doc_len, d):
        if self.nl>1:
            sum_d = doc_len + self.l_doc_lengths[other_l, d] + 2*self.alpha_sum -1
        else:
            sum_d = doc_len + self.alpha_sum

        return sum_d

    cdef int[:] get_scalarK(self, Py_ssize_t l, long[:] topics_to_eval) except *:
        cdef int[:] scalarK
        cdef int nwords_k

        cdef double min_nwords, max_nwords, denom
        #scalarK = np.zeros(len(topics_to_eval), dtype=np.intc)
        scalarK = np.zeros(self.ntopics, dtype=np.intc)
        #renormalise to this range. 
        # nu = self.dof0_var + nwords_k - self.ndim + 1
        # our scalar is from 1 to 30.
        # So the largest nwords_k will have scalar=30
        # The smallest nwords_k will have scalar=1
        # the resulting nu will be just 1 + scalar
        min_nwords = float('inf')
        max_nwords = -float('inf')

        for k in topics_to_eval:
            nwords_k = self.get_nwords(l, k)
            min_nwords = min(min_nwords, nwords_k)
            max_nwords = max(max_nwords, nwords_k)

        denom = (max_nwords - min_nwords)# * 29 + 1)
        
        for k in topics_to_eval:
            nwords_k = self.get_nwords(l, k)
            if max_nwords==min_nwords:
                scalar=30
            else:
            #    print(nwords_k, min_nwords, denom)
                scalar = ((nwords_k-min_nwords)/denom)*29 + 1

            scalarK[k] = int(scalar)

        return scalarK



    cdef void sample_one_doc(self, Py_ssize_t l, Py_ssize_t other_l, Py_ssize_t d, int[:] doc_vec, int doc_len) except *:

        cdef object lang, posterior
        cdef int nwords_k, scalar
        cdef Py_ssize_t k, w, ix, old_k, new_k
        cdef int[:] doc_position_topic, scalarK
        cdef np.ndarray[DTYPE_t, ndim=1] word_emb, euc_dist, scores
        cdef long[:] topics_to_eval
        cdef np.ndarray[DTYPE_t, ndim=2] mu_allk, emb_dist_allk
        cdef double[:] topic_term_scores, p_zk_d_allk, log_p_w_zk_allkG, p_w_zk_allk, p_w_zk_allkG
        cdef double topic_score, sum_scores, log_max, p_zk_d, log_p_w_zkG, log_score, sample, \
        sum_d, p_w_zk, Z, ZG, p_w_zkG, p_w_zkD, intp

        cdef np.intp_t i, j

        topic_term_scores = np.zeros(self.ntopics, dtype=DTYPE)
        p_zk_d_allk = np.zeros(self.ntopics, dtype=DTYPE)
        scores = np.zeros(self.ntopics, dtype=DTYPE)

        p_w_zk_allk = np.zeros(self.ntopics, dtype=DTYPE)
        p_w_zk_allkG = np.zeros(self.ntopics, dtype=DTYPE)
        log_p_w_zk_allkG = np.zeros(self.ntopics, dtype=DTYPE)

        doc_position_topic = self.l_topicAssign_doc_word[l, d]
        lang = self.langs[l]

        sum_d = self.get_sum_d(other_l, doc_len, d)

        if self.shared_params == 100:
            pass 

        elif self.shared_params==1:
            posterior = self.l_posterior[0]
        else:
            posterior = self.l_posterior[l]

        #print(doc_len, "--", np.asarray(doc_vec))
        for w in range(doc_len):
            ix = doc_vec[w]
            old_k = doc_position_topic[w]
            if old_k==-99:
                continue
            
            # drop topic 
            self.l_word_topic_counts[l, ix, old_k] -= 1
            self.l_alltopic_counts[l, old_k] -= 1
            self.l_d_k[l, d, old_k] -= 1

            topic_score = 0.0
            sum_scores = 0.0
            log_max = -float('inf')

            # collect discrete probs for p_w_zk_allk and p_zk_d_allk
            # HANDLE DISCRETE PROBS

            for k in range(self.ntopics):
                
                p_w_zk = (self.beta+ self.l_word_topic_counts[l, ix, k]) \
                / (self.l_alltopic_counts[l, k] + self.l_beta)

                p_w_zk_allk[k] = p_w_zk
                if self.nl>1:
                    p_zk_d = (self.alpha_2k + self.l_d_k[l, d, k] + self.l_d_k[other_l, d, k]) / sum_d
                else:
                    p_zk_d = (self.alpha_k + self.l_d_k[l, d, k]) /sum_d
                p_zk_d_allk[k] = p_zk_d


            if self.shared_params==100:
                # Not Gaussian
                for k in range(self.ntopics):
                    topic_score = p_w_zk_allk[k] * p_zk_d_allk[k]
                    sum_scores += topic_score
                    topic_term_scores[k] = topic_score

            # CONTINUOUS PROBS
            else:
                # Gaussian
                nwords_k = self.get_nwords(l, old_k)      
                word_emb = lang.get_emb2(ix)
                posterior.update(old_k, word_emb, nwords_k, "drop")
                # First do a ranking by p_zk_d * (embedding distance). Only calculate the
                # top 20 topics, ignore the rest.
 
                mu_allk = posterior.get_mu_allk()
                emb_dist_allk = word_emb - mu_allk

                # Find number of topics to eval
                if self.ntopics<self.max_eval:
                    topics_to_eval = memoryview(np.array(range(self.ntopics)))
                else:
                    euc_dist = np.linalg.norm(emb_dist_allk, axis=1)
                    scores = 1/euc_dist * p_zk_d_allk * p_w_zk_allk
                    topics_to_eval = np.argsort(-scores)[:self.max_eval]


                if self.scaling_dof0==1:
                    scalarK = self.get_scalarK(l, topics_to_eval)
                
                for k in topics_to_eval:
                    nwords_k = self.get_nwords(l, k)
                    if self.scaling_dof0:
                        scalar = scalarK[k]
                        posterior.set_dof0_var((self.ndim - nwords_k + scalar))

                    log_p_w_zkG = posterior.calc_mv_tdens(k, nwords_k, emb_dist_allk[k])
                    log_p_w_zk_allkG[k] = log_p_w_zkG
                    # check how important this is
                    if log_p_w_zkG > log_max:
                        log_max = log_p_w_zkG

                # normalise
                Z = 0.0
                ZG = 0.0
                # Sum for normalising constant 
                for k in topics_to_eval:
                    p_w_zkG = exp(log_p_w_zk_allkG[k]-log_max)
                    ZG += p_w_zkG
                    p_w_zk_allkG[k] = p_w_zkG
                    Z += p_w_zk_allk[k]

                for k in topics_to_eval:
                    if self.interpolate==0:
                        # purely gaussian
                        # dont interpolate with discrete
                        p_w_zk = p_w_zk_allkG[k]/ZG
                    else:
                #        nwords_k = self.get_nwords(l, k)
                        if self.interpolate<=1:
                            intp = self.interpolate

                        elif self.interpolate==2:
                            intp = np.random.beta(self.langs[l].vocab_prior[ix]\
                                + self.l_word_topic_counts[l, ix, k] + 1/self.l_intp_prior[l],\
                                    posterior.get_betab_allk()[k]/self.nl + 0.01)
                        else:
                            sys.exit("invalid interpolate value", self.interpolate)


                        p_w_zkG = (1-intp) * (p_w_zk_allkG[k]/ZG) 
                        p_w_zkD = (intp * (p_w_zk_allk[k]/Z))
                        p_w_zk = p_w_zkG + p_w_zkD

                    topic_score = (p_w_zk * p_zk_d_allk[k])
                    sum_scores += topic_score
                    topic_term_scores[k] = topic_score
            
            #print(np.asarray(topic_term_scores))
            new_k = self.sample_topic(sum_scores, topic_term_scores)

            doc_position_topic[w] = new_k
            self.l_word_topic_counts[l, ix, new_k] += 1
            self.l_alltopic_counts[l, new_k] += 1
            self.l_d_k[l, d, new_k] += 1
            if self.shared_params!=100 and self.shared_params!=99:
                nwords_k = self.get_nwords(l, new_k) 
                posterior.update(new_k, word_emb, nwords_k, "add")


        self.l_topicAssign_doc_word[l, d] = doc_position_topic

    cdef Py_ssize_t predict_oov_word_topic(self, Py_ssize_t l, str word):

        #cdef np.ndarray[DTYPE_t, ndim=1] word_emb
        cdef Py_ssize_t k, log_maxtopic


        if word in self.oov_word_topic:
            return self.oov_word_topic[word]

        if self.shared_params==1:
            posterior = self.l_posterior[0]
        else:
            posterior = self.l_posterior[l]
        
        lang = self.langs[l]
        

        if word in lang.embedding.vocab:
            word_emb = lang.embedding.get_vector(word)
            #ix = lang.w2ix[word]
            #word_emb = lang.get_emb2(ix)
        else:
            return -99
        
        #mu_allk = posterior.get_mu_allk()
        #emb_dist_allk = word_emb - mu_allk

        log_max = -float('inf') 
        log_maxtopic = 0

        for k in range(self.ntopics):
            #log_p_w_zkG = posterior.calc_mv_tdens(k, nwords_k, emb_dist_allk[k], self.lw_alpha)
            log_p_w_zkG = posterior.calc_mv_ndens(k, word_emb)

            if log_p_w_zkG > log_max:
                log_max = log_p_w_zkG
                log_maxtopic = k

        #print("oov:", word, "maxtopic:", log_maxtopic)
        self.oov_word_topic[word] = log_maxtopic

        return log_maxtopic


    cpdef Py_ssize_t sample_topic(self, double sum_scores, double[:] topic_term_scores) except *:

        cdef double sample
        cdef double r = rand()
        cdef Py_ssize_t new_k

        #sample = random.random() * sum_scores
        sample = r/RAND_MAX * sum_scores
        new_k = -1
        
        while sample > 0.0:
            new_k += 1
            sample -= topic_term_scores[new_k]

        if new_k == -1:
            print(np.array(topic_term_scores))
            raise ValueError("New topic not sampled!")

        return new_k


    def model_loglikelihood(self, test_data):
        # Implementation from Mallet 
        # we may only want the log likelihood over the test set.
        llh = 0.0
        
        # Do documents first
        topic_counts = np.zeros(self.ntopics)
        topic_log_gammas = np.zeros(self.ntopics)
        doc_topics = []

        for k in range(self.ntopics):
            topic_log_gammas[k] = loggamma(self.alpha[k])

        ##########################
        # this whole bunch is added
        vocab_d = {}
        for l in range(self.nl):
            test_data = test_data[l]
            test_data, _ = self.langs[l].prep_data(test_data)
            test_data = self.langs[l].filter_oov_embeddings(test_data, test_mode=True)
            words = [set(d) for d in test_data]
            words = set.union(*words)
            vocab_d[l] = words

            lang = self.langs[l]

            self.update_corpus_index(test_data)
            self.assign_topics(l, lang, test_data, mode="test")
            total_n = self.langs[l].ndocs + len(test_data)
            ndocs = self.langs[l].ndocs

        for d in range(ndocs, total_n):

        ########################

        #for d in range(len(self.l_doc_vecs[0])):
            for l in range(self.nl):
                doc_vec = self.l_doc_vecs[l, d]
                doc_len = self.l_doc_lengths[l, d]
                topic_counts = self.l_d_k[l, d]

                for k in range(self.ntopics):
                    if topic_counts[k]>0:
                        llh += (loggamma((self.alpha[k]+topic_counts[k])) \
                    - topic_log_gammas[k])
                llh -= loggamma(self.alpha_sum + doc_len)

        # Add the parameter sum term
        #llh += (len(self.langs[0].train_data) * loggamma(self.alpha_sum))
        llh += (len(test_data) * loggamma(self.alpha_sum))
        # And the topics

        log_gamma_beta = loggamma(self.beta)

        for l in range(self.nl):
            # we only want testix not trainix

            for ix in self.langs[l].ix2w.keys():
                word = self.langs[l].ix2w[ix]
                if word not in vocab_d[l]:
                    continue
                else:
                    pass

                topic_counts = self.l_word_topic_counts[l, ix]

                for k in range(self.ntopics):
                    if topic_counts[k] ==0:
                        continue
                    llh += (loggamma(self.beta + topic_counts[k]) - log_gamma_beta)

                if np.isnan(llh):
                    raise ValueError("Nan after topic:", k)

            for k in range(self.ntopics):
                llh -= loggamma( (self.beta * self.langs[l].vocab_size) + self.l_alltopic_counts[l, k])
#
            llh += (self.ntopics * loggamma((self.beta*self.langs[l].vocab_size)))
            if np.isnan(llh):
                raise ValueError("Nan at the end")

        return llh


    def print_topic_stats(self):
        for l in range(self.nl):
            if self.shared_params==1:
                el=0
            else:
                el=l
            print("language:", l)
            print("\nTotal words:", np.asarray(self.l_alltopic_counts[l]).sum())
            for k in range(self.ntopics):
                print("topic:", k)
                print("mean;", np.array(self.l_posterior[el].get_mu_allk()))
        #        print("mean:", np.array(self.l_posterior[l].mu_allk[k]))
        #        print("cov:", np.array(self.l_posterior[l].cov_allk[k]))
                print("upperT:", np.array(self.l_posterior[el].get_upperT_allk())[k])
                print("nwords:", self.l_alltopic_counts[l][k])


    def save(self, pickle_dir, suffix=""):
        save_name = os.path.join(pickle_dir, suffix)
        print("saving to:", save_name)

        # first save the counts
        save1 = save_name + "_l_word_topic_counts"
        pickle.dump(self.l_word_topic_counts.base, open(save1, 'wb'))

        save2 = save_name + "_l_alltopic_counts"
        pickle.dump(self.l_alltopic_counts.base, open(save2, 'wb'))

        save3 = save_name + "_l_d_k"
        pickle.dump(self.l_d_k.base, open(save3, 'wb'))

        # then save the dicts:
        save4 = save_name + "_l_topicAssignment"
        pickle.dump(self.l_topicAssign_doc_word.base, open(save4, 'wb'))

        save5 = save_name + "_l_doc_vecs"
        pickle.dump(self.l_doc_vecs.base, open(save5, 'wb'))

        for l in range(self.nl):
            if self.shared_params == 100:
                continue 

            if self.shared_params==1:
                self.l_posterior[0].save(pickle_dir, suffix+str(l))
                break
            else:
                self.l_posterior[l].save(pickle_dir, suffix+str(l))

        save_name = os.path.join(pickle_dir, suffix+"rs")
        pickle.dump(random.getstate(), open(save_name, 'wb'))

    cpdef load(self, pickle_dir, suffix=""):

        load_name = os.path.join(pickle_dir, suffix)

        load1 = load_name + "_l_word_topic_counts"
        self.l_word_topic_counts = pickle.load(open(load1, 'rb'))
        #print(np.asarray(self.l_word_topic_counts[0,:,:]).sum(axis=0))
        #print(np.asarray(self.l_word_topic_counts[1,:,:]).sum(axis=0))

        load2 = load_name + "_l_alltopic_counts"
        self.l_alltopic_counts = pickle.load(open(load2, 'rb'))

        load3 = load_name + "_l_d_k"
        self.l_d_k = pickle.load(open(load3, 'rb'))

        load4 = load_name + "_l_topicAssignment"
        self.l_topicAssign_doc_word = pickle.load(open(load4, 'rb'))

        load5 = load_name + "_l_doc_vecs"
        self.l_doc_vecs = pickle.load(open(load5, 'rb'))


        for l in range(self.nl):
            print("getting doc lengths for language:", self.langs[l].name)
            
            train_data = self.langs[l].train_data
            for d in range(len(train_data)):
                words = train_data[d]
                self.l_doc_lengths[l, d] = len(words)

            if self.prior_mean==1:
                print("use sample mean")
                mean = self.langs[l].get_sample_mean()
            else:
                mean = np.zeros(self.ndim)

            if self.cov0_scalar>0: 
                print("Use empirical covariance")
                sample_cov = np.identity(self.ndim)*self.cov0_scalar 
            else:
                sample_cov = self.langs[l].get_sample_cov()

            self.l_intp_prior[l] = self.langs[l].vocab_size/self.langs[0].ndocs

        for l in range(self.nl):
            if self.shared_params==1:
                self.l_posterior[0] = Posterior(self.ntopics, self.ndim, mu0=mean, mu0_b=self.mu0_b, sigma0=sample_cov)
                self.l_posterior[0].load(pickle_dir, suffix+str(l))
                break
            else:
                self.l_posterior[l] = Posterior(self.ntopics, self.ndim, mu0=mean, mu0_b=self.mu0_b, sigma0=sample_cov)
                print("load posterior")
                #self.l_posterior[l].load(pickle_dir, suffix+str(l))

            print(pickle_dir, suffix+str(l))
        
        load_name = os.path.join(pickle_dir, suffix+"rs")
        state = pickle.load(open(load_name, 'rb'))
        random.setstate(state)
        
        #self.print_topic_stats()
