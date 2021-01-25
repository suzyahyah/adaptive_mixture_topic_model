#!/usr/bin/python
# Author: Suzanna Sia
import yaml
import pandas as pd
import numpy as np
import os
import pdb
import time
import requests



def getData(dp, l1l2, mode=""):

    if len(mode)==0:
        raise Exception("Must have train, valid or test mode")

    with open(dp, 'r') as f:
        files = yaml.load(f)

    if l1l2.startswith('en'):
        l1="en"
        l2=l1l2[-2:]

        f1 = files['parallel_en{}'.format(l2)]
        f2 = files['parallel_{}'.format(l2)]

        t1 = files['en_docs']
        embed1 = files['en_multi_embed']
        embed2 = files['{}_multi_embed'.format(l2)]

    if mode=="train":
        ids1, data1 = read_data(f1 + ".train")
        ids2, data2 = read_data(f2 + ".train")

    if mode=="test" or mode=="valid":
        ids1, data1 = read_data(t1 + ".{}".format(mode))
        ids2, data2 = read_data(f2 + ".{}".format(mode))

    return ids1, data1, ids2, data2

def getEmbed(dp, l1l2):
    with open(dp, 'r') as f:
        files = yaml.load(f)

    if l1l2.startswith('en'):
        l1="en"
        l2=l1l2[-2:]

        embed1 = files['en_multi_embed']
        embed2 = files['{}_multi_embed'.format(l2)]

    return embed1, embed2


def read_data(txt):
    data = pd.read_csv(txt, sep="\t", header=None, names=['query_id', 'topic', 'sentence'], encoding='utf-8')
    text_data = data.iloc[:, 2].values
    doc_ids = data.iloc[:, 0].values
    return doc_ids, text_data


def getLangs(c, data, ndocs=0, ndim=300, datapath=""):

    if args.c==1:
        from language_c2 import Language
    else:
        from language import Language

    languages = []

    if args.data=="gen":

        data_folder="data/generatedData/ndocs{}".format(args.ndocs)
        embeddings_folder = "assets/embeddings/ndocs{}".format(args.ndocs)

        _, train_a = read_data(os.path.join(data_folder, 'allgendocs_a.txt.train'))
        docids_a, test_a = read_data(os.path.join(data_folder, 'allgendocs_a.txt.test'))
        lang1 = Language("a", args.ndim, train_a, os.path.join(embeddings_folder, "gen_a.vec"))

        _, train_b = read_data(os.path.join(data_folder, 'allgendocs_b.txt.train'))
        docids_b, test_b = read_data(os.path.join(data_folder, 'allgendocs_b.txt.test'))
        lang2 = Language("b", args.ndim, train_b, os.path.join(embeddings_folder, "gen_b.vec"))

    else:
        _, train_1, _, train_2 = getData(args.data_path, args.data, mode="train")
        test_ids1, test_1, test_ids2, test_2 = getData(args.data_path, args.data, mode="test")
        valid_ids1, valid_1, valid_ids2, valid_2 = getData(args.data_path, args.data, mode="valid")

        embed1, embed2 = getEmbed(args.data_path, args.data)

        if args.dev==1:
            print("Subsampling data")
            train_1 = train_1[:1005]
            train_2 = train_2[:1005]

        # assume num_lang==2
        lang1 = Language(l1, args.ndim, train_1, embed1)
        lang2 = Language(l2, args.ndim, train_2, embed2)

    languages.append(lang1)
    languages.append(lang2)
    for l in range(len(languages)):
        languages[l].load_embeddings(filter=True)
 
    return languages

def get_latest_iter(pickle_dir):
    pfiles = os.listdir(pickle_dir)
    pfiles = [f for f in pfiles if f.endswith("l_d_k")]
    pfiles = [(''.join(c for c in f if c.isdigit())) for f in pfiles]
    load_iter = max([int(i) for i in pfiles])
    print(pickle_dir, "; MAX ITER:", load_iter)
    return load_iter

#if __name__=="__main__":
#    wordsout_dir = "results/top_words/dev2/ne-split-gauss-ntopics20/stagger0-sharedParams100-temp-interpolate0-scaling1-trainsize1000.split0"
#    external_score_top_words(wordsout_dir, score_type="npmi")

