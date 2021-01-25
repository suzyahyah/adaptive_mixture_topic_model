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
parser.add_argument('--concat', action="store_true", help="use subsample data")

args = parser.parse_args()

def init():

    if args.full:
        print("> Experimental mode, using full dataset..")
        data_path = "./configs/data_paths.yml"

    else:
        print("test mode, using subsample data..")
        data_path = "./configs/data_paths_subsample.yml"

    # load data from configs

    with open(data_path, 'r') as f:
        files = yaml.load(f)
    

    query2sw = pd.read_csv(files['en_sw_rel'], sep="\t", header=None, names=['en_query_id',
    'sw_doc_id', 'rel'])
    query2doc = pd.read_csv(files['en_query_doc_rel'], sep="\t", header=None,
    names=['en_query_id', 'en_doc_id', 'rel'])

    if args.concat:

        sw_docs = pd.read_csv(files['swahili_docs_tra'], sep="\t", header=None, names=['sw_doc_id', 'title', 'content'])
    
    else:
        sw_docs = pd.read_csv(files['swahili_docs_raw'], sep="\t", header=None,
        names=['sw_doc_id', 'title', 'content'])

    for mode in ['.train', '.test']:
        en_docs = pd.read_csv(files['en_docs']+mode, sep="\t", header=None, names=['en_doc_id', 'title', 'content'])

        toWriteEN = []
        toWriteSW = []
        
        for query_id in query2sw['en_query_id'].unique():
            try:
                sw_doc_id = query2sw[query2sw['en_query_id']==query_id]['sw_doc_id'].values[0]
                sw_content = sw_docs[sw_docs['sw_doc_id']==sw_doc_id]['content'].values[0]
                sw_title = sw_docs[sw_docs['sw_doc_id']==sw_doc_id]['title'].values[0]
            except:
                continue

            match_docs = query2doc[query2doc['en_query_id']==query_id]
            match_docs = match_docs[match_docs['rel']==1]
            # 2 is exact match, 1 is inexact match

            for en_doc_id in match_docs['en_doc_id']:
                en_doc = en_docs[en_docs['en_doc_id']==en_doc_id]

                if len(en_doc)==0:
                    continue

                en_title = en_doc['title'].values[0]
                en_content = en_doc['content'].values[0]

                if args.concat:
                    contents = en_content + " " + sw_content
                    toWriteEN.append([str(en_doc_id), en_title, contents])
                else:
                    toWriteEN.append([str(en_doc_id), en_title, en_content])
                    toWriteSW.append([str(sw_doc_id), sw_title, sw_content])

        toWriteEN = pd.DataFrame(toWriteEN)
        toWriteSW = pd.DataFrame(toWriteSW)

        if args.concat:
            toWriteEN.to_csv(files['concat_en_sw']+mode, sep="\t", header=None, index=False)
        else:
            toWriteEN.to_csv(files['parallel_en']+mode, sep="\t", header=None, index=False)
            toWriteSW.to_csv(files['parallel_sw']+mode, sep="\t", header=None, index=False)

if __name__ == "__main__":

    init()
