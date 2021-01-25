#!/usr/bin/python
# Author: Suzanna Sia
import pdb
import pickle
import os
import numpy as np

PICKLEDIR = os.path.dirname(os.path.realpath(__file__))


def get_max_vocab_size(langs):

    if len(langs)==1:
        vocab_size = langs[0].vocab_size
    else:
        all_vocab_sizes = []

        for l in langs:
            all_vocab_sizes.append(l.vocab_size)

        vocab_size = max(all_vocab_sizes)


    return vocab_size


def calc_topic_proportions(ntopics, topicAssignment_doc_word, doc_lengths):
    # returns immediate topic proportions

    all_topic_probs = []
    invalid_doc_ix = []

    for d in range(len(topicAssignment_doc_word)):

        topic_counts = np.zeros(ntopics)
        doc_position_topic = topicAssignment_doc_word[d]
        doc_len = doc_lengths[d]
        invalid = 0

        if d<5:
            print(doc_len, np.asarray(doc_position_topic))

        for w in range(doc_len):
            topic = doc_position_topic[w]

            if topic==-99:
                # get embedding
                # filler assigned to out of vocab topic
                invalid += 1
                continue

            topic_counts[topic] += 1

        # normalize counts
        valid_len = doc_len - invalid

        if valid_len==0:
            invalid_doc_ix.append(d)
            # this is correct, topic prop should be 0
            topic_probs = np.zeros(ntopics)
        else:
            topic_probs = topic_counts/valid_len

        all_topic_probs.append(topic_probs)

    return all_topic_probs, invalid_doc_ix
