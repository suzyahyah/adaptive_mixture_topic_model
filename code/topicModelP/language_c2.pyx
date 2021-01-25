#!/usr/bin/python
# Author: Suzanna Sia
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: infertypes=True
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

from preprocess import Preprocessor
import gensim
import time
from gensim.models import KeyedVectors
from collections import Counter
import itertools
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np
cimport numpy as np
import time
import os

import pp1

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class Language():

    cdef public str name, embed_path
    cdef public int ndim, ndocs, vocab_size, max_length
    cdef public list train_data, test_data, vocab, doc_vecs
    cdef public dict ix2w, w2ix, oov_w2ix, oov_ix2w, temp_embedding
    cdef public object counts
    cdef public object embedding
    cdef public int[:] doc_lengths
    cdef public double[:] vocab_prior
    cdef public int[:,:] count_matrix
    cdef public int[:] word_doc_counts
    cdef np.ndarray embedding2

    def __init__(self, name, ndim, train_data, embed_path, ntopics, vocab_fn=""):

        self.name = name
        self.ndim = ndim
        self.train_data, self.counts = self.prep_data(train_data)
        stopwords = set(line.strip() for line in open('assets/stoplists/en.txt'))
        _, _, self.train_data = pp1.create_vocab(stopwords, train_data, process=True)
        self.test_data = []
        self.ndocs = len(train_data)
        self.vocab = []
        self.vocab_size = 0
        self.ix2w = {}
        self.w2ix = {}
        self.oov_w2ix = {}
        self.oov_ix2w = {}
        self.doc_vecs = []
        # embedding
        self.embed_path = embed_path
        self.embedding = None
        self.temp_embedding = {}
        self.vocab_prior = np.zeros(self.vocab_size, dtype=DTYPE)
        self.get_vocab(ntopics, vocab_fn)
        self.doc_lengths = np.zeros(self.ndocs, dtype=np.intc)
        self.max_length = 0

        self.embedding2 = np.zeros((self.vocab_size, self.ndim), dtype=DTYPE)

        # counts

        # matrix counts
        self.word_doc_counts, self.count_matrix = self.make_vocab_matrix()

    def make_vocab_matrix(self):

        count_matrix = np.zeros((len(self.w2ix), len(self.w2ix)), np.intc)
        word_doc_counts = np.zeros(len(self.w2ix), np.intc)
        start = time.time()
        cv = CountVectorizer(vocabulary=self.w2ix)
        train_data = [" ".join(d) for d in self.train_data]
        doc_vocab_counts = cv.fit_transform(train_data)


        for doc in range(doc_vocab_counts.shape[0]):
            nonzero_ixs = np.nonzero(doc_vocab_counts[doc])[1]
            for ix in nonzero_ixs:
                word_doc_counts[ix] += 1

            update_pairs = list(itertools.combinations(nonzero_ixs, 2))
            for pair in update_pairs:
                # keep it one-sided
                if pair[0]>pair[1]:
                    count_matrix[pair[1], pair[0]]+=1
                else:
                    count_matrix[pair[0], pair[1]]+=1

        print("constructed matrix:", time.time() -start)
        return word_doc_counts, count_matrix


    def write_vocab(self, vocab_fn=""):

        words = [self.ix2w[i] for i in range(len(self.ix2w))]
        towrite = "\n".join(words)

        with open(vocab_fn, 'w') as f:
            f.write(towrite)
        print("written vocab to:", vocab_fn)
    
    def get_vocab(self,ntopics, vocab_fn):
        if len(vocab_fn)!=0:
            with open(vocab_fn, 'r') as f:
                vocab = f.readlines()
            vocab = [v.strip() for v in vocab]
            self.w2ix = {vocab[i]:i for i in range(len(vocab))}
            self.ix2w = {i:vocab[i] for i in range(len(vocab))}
            self.vocab = vocab
            self.vocab_size = len(vocab)
            print(f"Language:{self.name} from:{vocab_fn} ndocs:{len(self.train_data)}")

        else:
            words = [set(d) for d in self.train_data]
            words = set.union(*words)
            vocab = sorted(list(words))

            self.w2ix = {vocab[i]:i for i in range(len(vocab))}
            self.ix2w = {i:vocab[i] for i in range(len(vocab))}
            self.vocab = vocab
            self.vocab_size = len(vocab)

            print("Language:", self.name)
            print("No. of train documents:", len(self.train_data))
            print("Vocab size:", self.vocab_size)

            self.vocab_prior = np.zeros(len(self.w2ix))
            for word in self.w2ix:
                ix = self.w2ix[word]
                self.vocab_prior[ix] = self.counts[word]/ntopics



    def prep_data(self, train_data, min_count=5, excl=10):
    #def prep_data(self, train_data, min_count=1, excl=0):

        preprocessor = Preprocessor(self.name)
        print(self.name, "filter words with min count:", min_count)
        original = train_data[5]

        train_data = [preprocessor.clean(d) for d in train_data]

        data = []
        for doc in train_data:
            doc = doc.lower().split()
    #        if len(doc)>200:
    #            doc = doc[:200]
    #        else:
    #            pass
            data.append(doc)

        train_data = data
        train_data = [preprocessor.rem_stopwords(d) for d in train_data]
        # word has to appear at least x times in the corpus
        counts_data = [d for doc in train_data for d in doc]
        counts = Counter(counts_data)
        
        if excl>0:
            top_excl = counts.most_common(excl)
            top_excl = [w[0] for w in top_excl]
        
            print("excl most common words:", " ".join(top_excl).encode('utf-8'))
            train_data = [[w for w in doc if counts[w]>=min_count and w not in top_excl] for doc in train_data]
        else:
            train_data = [[w for w in doc if counts[w]>=min_count] for doc in train_data]

       # need to deal with repeated words covariance...
        #unique = True
        #if unique:
        #    train_data = [list(set(d)) for d in train_data]
        print("Sanity check:\n")
        print("original:", original)
        print("process:", " ".join(train_data[5]))
        return train_data, counts

    def load_embeddings(self, filter=True):

        start = time.time()
        if os.path.exists(self.embed_path+".slim"):
            print("Loading slim embeddings", end=" ")
            self.embedding = KeyedVectors.load_word2vec_format(self.embed_path+".slim")
            
        else:
            print("Warning! No slim embeddings.. Generate them first")
            self.embedding = KeyedVectors.load_word2vec_format(self.embed_path, limit=500000)   

        print("Loaded embeddings..", time.time() - start)

        if filter:
            self.train_data = self.filter_oov_embeddings(self.train_data)


    def filter_oov_embeddings(self, data, test_mode=False):
        docs = []
        max_length = 0
        
        i = 0
        
        for d, words in enumerate(data):
            doc = []
            oov_count = 0
            for w, word in enumerate(words):
                if word in self.embedding.vocab:
                    doc.append(word)
                    if word not in self.w2ix:
                        oov_count += 1

                        if word not in self.oov_w2ix:
                            # give it a new index
                            ix = self.vocab_size + i
                            self.oov_w2ix[word] = ix
                            self.oov_ix2w[ix] = word
                            i+=1
                    else:
                        ix = self.w2ix[word]
                        self.embedding2[ix] = np.asarray(self.embedding.get_vector(word),\
                                dtype=DTYPE)
                else:
                    pass

            if len(doc)>max_length:
                max_length = len(doc)
            if len(doc)==oov_count and oov_count>0:
                print("doc contains all oov. pass", words)

            docs.append(doc)
        # dont update the max length if in test mode
        if not test_mode:    
            self.max_length = max_length

        print("max length of docs:", self.max_length)
        print("oov words:", (" ".join(self.oov_w2ix.keys())).encode())
        return docs

    def get_doc_lengths(self):
        doc_lengths = np.array([len(self.train_data[i]) for i in range(len(self.train_data))])
        return doc_lengths

    cpdef np.ndarray[DTYPE_t, ndim=1] get_emb2(self, Py_ssize_t ix):
        return self.embedding2[ix]

    def get_emb(self, word, k=-1):

        if word in self.embedding.vocab:
            return self.embedding.get_vector(word)
        #else:
        #    if k!=-1:
        #        mu = self.l_posterior[l].mu_allk[k]
                #upperT = self.posterior.upperT_allk[k]
       #         upperT = self.upperT_0
       ##         z = np.random.normal(size=self.ndim)
       #         word_emb = mu+np.matmul(upperT.T, z)
       #         self.temp_embedding[word] = word_emb

#        else:
#            if word not in self.temp_embedding:
#                word_emb = np.random.uniform(-0.01, 0.01, self.ndim)
#                self.temp_embedding[word] = word_emb
#            else:
#                word_emb = self.temp_embedding[word]
#        return word_emb

    def get_sample_cov(self):
        all_embed = []
        for word in self.embedding.vocab:
            word_emb = self.get_emb(word)
            all_embed.append(word_emb)

        cov = np.cov(np.array(all_embed), rowvar=False)
        return cov
            
 
    def get_sample_mean(self):
        total_emb = np.zeros(self.ndim)
        for word in self.embedding.vocab:
            word_emb = self.get_emb(word)
            total_emb += word_emb
        sample_mean = total_emb/len(self.embedding.vocab)
        return sample_mean

