#!/usr/bin/python
# Author: Suzanna Sia
import string
import numpy as np
import pdb
import argparse
import pandas as pd
import gensim
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument('--full', action="store_true", help="use subsample data")

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

   
    docs = pd.read_csv(files['concat_en_sw']+".train", header=None, sep="\t", names=['doc_id',
        'title', 'content'])
    
    model = gensim.models.KeyedVectors.load_word2vec_format(files['en_embed'], limit=30000)
    docs['content_'] = docs['content'].apply(lambda x: convert_string2ix(x, model))

    docs['content_'].to_csv(files['concat_en_sw']+".train.vec", index=False)

    #pd.to_csv(docs, header=None, files['concat_en_sw']+".train.vec")

    #preprocess to remove sw, punct and lower case.
    
    #docs.apply(lambda x: model.index2entity.index[


def convert_string2ix(sentence, model):
    
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator).lower()
    sentence = sentence.split()
    
    ixs = []

    for word in sentence:
        try:
            ixs.append(model.index2entity.index(word)+1) # because the first line is not a vec
        except:
            #print("Word not found:", word)
            pass

    ixs = " ".join([str(x) for x in ixs])
    return ixs



   #for mode in ['.train', '.test']:
        
        # files['concat_en_sw']
        # files['parallel_en']
        # files['parallel_sw']
        #
        

if __name__ == "__main__":

    init()
