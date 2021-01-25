#!/usr/bin/python
# Author: Suzanna Sia
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import yaml
import pandas as pd
from language import Language
from sklearn.datasets import fetch_20newsgroups

def wiki_dataset():
    data_path = "./configs/data_paths.yml"
    with open(data_path, 'r') as f:
        files = yaml.load(f)

    ################
    all_en = []
    #for suffix in ['fi', 'ro', 'de', 'fr']:
    for suffix in ['pl', 'tr', 'sv', 'es', 'fi', 'ro', 'de', 'fr']:
        print(suffix, end=" ")
        all_data = []
        file1 = files['parallel_{}'.format(suffix)]
        file_en = files['parallel_en{}'.format(suffix)]

        _, train_data = read_data(file1 + ".train")
        _, test_data = read_data(file1 + ".test")
        _, valid_data = read_data(file1 + ".valid")

        _, en_train_data = read_data(file_en + ".train")
        all_en.extend(en_train_data)

        all_data.extend(train_data)
        all_data.extend(test_data)
        all_data.extend(valid_data)

        embed = files['{}_multi_embed'.format(suffix)]
        ndim =300
        lang = Language(suffix, 300, all_data, embed)
        lang.load_embeddings(filter=True)

    # english
    _, test_en = read_data(files['en_docs']+".test")
    _, valid_en = read_data(files['en_docs']+".valid")

    all_en.extend(test_en)
    all_en.extend(valid_en)

    embed = files['en_multi_embed']
    ndim = 300
    lang = Language("en", 300, all_en, embed)
    lang.load_embeddings(filter=True)


def news_dataset():
    print("Loading 20 news group..")
    data_folder="./data"
    train_data = fetch_20newsgroups(data_home=data_folder, subset="train",
    remove=('headers', 'footers', 'quotes'))
    test_data = fetch_20newsgroups(data_home=data_folder, subset='test', remove=('headers',
        'footers', 'quotes'))

    train_data = [d.replace('\n', ' ') for d in train_data['data']]
    test_data = [d.replace('\n', ' ') for d in test_data['data']]
    train_data.extend(test_data)

    embed_path = "assets/embeddings/wiki.multi.en.vec"

    lang = Language("en", 300, train_data, embed_path)
    lang.load_embeddings(filter=True)

   


def read_data(txt):
    data = pd.read_csv(txt, sep="\t", header=None, names=['query_id', 'topic', 'sentence'])
    text_data = data.iloc[:, 2].values
    doc_ids = data.iloc[:, 0].values
    return doc_ids, text_data

if __name__=="__main__":
    if sys.argv[1]=="wiki":
        wiki_dataset()
    elif sys.argv[1]=="news":
        news_dataset()
