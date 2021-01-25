#!/usr/bin/python
# Author: Suzanna Sia

import numpy as np
import pdb
import argparse
import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--full', action="store_true", help="use subsample data")
args = parser.parse_args()

def init():

    # load data from configs
    if args.full:
        print("> Experimental mode, using full dataset..")
        data_path = "./configs/data_paths.yml"
    else:
        print("test mode, using subsample data..")
        data_path = "./configs/data_paths_subsample.yml"

    with open(data_path, 'r') as f:
        files = yaml.load(f)

    
    # splits are performed on en_query_docs
    
    en_queries = pd.read_csv(files['en_queries'], sep="\t", header=None, names=['en_query_id',
        'title', 'title_sentence'])

    en_queries['title_sentence'].fillna(en_queries['title'], inplace=True)

    print("> Splitting train-test (0.8-0.2)")
    # tempMagic numbers
    train_queries, test_queries = train_test_split(en_queries, test_size=0.2, random_state=1)

    # Queries split
    train_queries.to_csv(files['en_queries']+".train", sep="\t", header=None, index=False)
    test_queries.to_csv(files['en_queries']+".test", sep="\t", header=None, index=False)


    # Docs split

    en2doc = pd.read_csv(files['en_query_doc_rel'], sep="\t", header=None,
            names=['en_query_id', 'en_doc_id', 'rel'])

    train_docs_id = en2doc[en2doc['en_query_id'].isin(train_queries['en_query_id'])]['en_doc_id']
    test_docs_id = en2doc[en2doc['en_query_id'].isin(test_queries['en_query_id'])]['en_doc_id']
   
    en_docs = pd.read_csv(files["en_docs"], sep="\t", header=None, names=['en_doc_id', 'title', 'title_sentence'])

    train_en_docs = en_docs[en_docs['en_doc_id'].isin(train_docs_id)]
    test_en_docs = en_docs[en_docs['en_doc_id'].isin(test_docs_id)]

    print("len train en;", len(train_en_docs))
    print("len test en:", len(test_en_docs))

    train_en_docs.to_csv(files['en_docs']+".train", sep="\t", header=None, index=False)
    test_en_docs.to_csv(files['en_docs']+".test", sep="\t", header=None, index=False)


if __name__ == "__main__":

    init()
