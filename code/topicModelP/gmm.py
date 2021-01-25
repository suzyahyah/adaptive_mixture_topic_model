#!/usr/bin/python
# Author: Suzanna Sia
# write GMM code here!
# incorporate in main script
# fit predict. 
# jiayou!
import pdb
import sys
import numpy as np 
from collections import Counter
import scipy as sp

from sklearn.mixture import GaussianMixture 

class GMM():

    def __init__(self, languages, ntopics, ndim):

        self.langs = languages
        self.num_lang = len(languages)
        self.ntopics = ntopics
        self.ndim = ndim
        self.cluster_dict = {}
        self.gmm = None
    
    def initialise(self, l=0):

        self.sample_mean = self.langs[l].get_sample_mean()
        self.sample_cov = np.identity(self.ndim)

#        means_init = np.array([self.sample_mean.copy() for i in range(self.ntopics)])
#        precisions_init = np.array([self.sample_cov.copy() for i in range(self.ntopics)])
        self.gmm = GaussianMixture(n_components=self.ntopics, covariance_type="full",\
                random_state=1, verbose=1)

    def run_model(self, l=0):
        
        print("Run GMM on vocab words...")

        unique_words = set()
        
        nonunique_words = []
        for doc in self.langs[l].train_data:
            for word in doc:
                unique_words.add(word)
                nonunique_words.append(word)

        # each word is going to get assigned to 1 topic. how do we print the 'top words'. Top
        # words are words that appear many times in the dataset and are associated with that
        # topic. So we do a counter for each topic.
        
        # do the fitting over unique words
        X_words = sorted(list(unique_words))
        X_embeddings = np.array([self.langs[l].get_emb(word) for word in X_words])
        self.gmm.fit(X_embeddings)
        X_labels = self.gmm.predict(X_embeddings)
        X_words_labels = zip(X_labels, X_words)

        nonunique_words_c = Counter(nonunique_words)

        for k in range(self.ntopics):
            self.cluster_dict[k] = []

        for cluster, word in X_words_labels:
            word_count = nonunique_words_c[word]
            self.cluster_dict[cluster].append((word, word_count))

        # now get top words
        for k in range(self.ntopics):
            cluster = self.cluster_dict[k]


    def calc_cluster_overlap(self, lda_cluster_file):
        # open topwords file
        # need to work with top 20 words to be better.
        lda_cluster_file = lda_cluster_file.replace('gmm', 'gauss')
        print("Reading topics from:", lda_cluster_file)
        with open(lda_cluster_file, 'r') as f:
            topic_words = f.readlines()

        vanilla_cluster_file = lda_cluster_file.replace("sharedParams1-", "sharedParams100-")
        vanilla_cluster_file = vanilla_cluster_file.replace("interpolate0.5-", "interpolate0-")

        print("Reading topics from:", vanilla_cluster_file)
        with open(vanilla_cluster_file, 'r') as f:
            vanilla_topic_words = f.readlines()

        topic_words = [tw.strip().split() for tw in topic_words]
        vanilla_topic_words = [tw.strip().split() for tw in vanilla_topic_words]
        

        total = 0
        overlap_topics = []
        for i in range(len(topic_words)):
            lda_cluster_words = set(topic_words[i])

            topic_props = []
            vanilla_topic_props = []

            for k in range(self.ntopics):
                gmm_cluster_words = self.cluster_dict[k]
                gmm_cluster_words = set([word[0] for word in gmm_cluster_words])
                overlap = lda_cluster_words.intersection(gmm_cluster_words)
                topic_props.append(len(overlap))
                
                vanilla_cluster_words = set(vanilla_topic_words[k])
                vanilla_overlap = lda_cluster_words.intersection(vanilla_cluster_words)
                vanilla_topic_props.append(len(vanilla_overlap))

            #topic_props
            # normalise
            
            if np.sum(topic_props)>0:
                topic_props = topic_props/np.sum(topic_props)
                max_topic = np.argmax(topic_props)

                vanilla_topic_props = vanilla_topic_props/np.sum(vanilla_topic_props)
                vanilla_max_topic = np.argmax(vanilla_topic_props)
                
                # check if there is overlap with lda clusters. if there is overlap then keep
                if topic_props[max_topic]>0.7:
                    if vanilla_topic_props[vanilla_max_topic]>0.4:
                        print("high gmm overlap but also high vanilla lda overlap.. So not adding")
                        print(" ".join(list(lda_cluster_words)))
                        continue

                    overlap_topics.append(i)
                    print("LDA-{} HIGH overlap with GMM-{}, props:{}".format(i, max_topic,\
                        topic_props[max_topic]))
                    print(" ".join(list(lda_cluster_words)))
                    print(" ")

                #elif topic_props[max_topic]<0.3:
                #    print("{} LOW overlap with topic:{}, props:{}".format(i, max_topic,\
                #topic_props[max_topic]))

            else:
                print("{} No overlap!".format(i)) 
                sys.exit(0)


        return overlap_topics


if __name__=="__main__":
    ndim=300
    ntopics=300
    ndocs=0
    datapath="./configs/data_paths.yml"
    c=1
    data = "en-de"
    l=0
    
    languages = utils.getLangs(c=c, data=data, ndocs=ndocs, ndim=ndim, datapath=datapath)
    gmm = GMM(languages, ntopics, ndim)
    gmm.initialise(l=l)
    gmm.run_model(l=l)

    # See 11540c4

