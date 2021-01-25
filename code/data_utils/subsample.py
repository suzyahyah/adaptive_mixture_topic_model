#!/usr/bin/python
# Author: Suzanna Sia

import yaml
import pandas as pd
import os
import pdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--full', action="store_true", help="subsample data")
parser.add_argument('--n', default=300, help="number to subsample")
args = parser.parse_args()

random_state = 0


#l1="sv"
#l2="tr"
#l3="pl"
#l4="es"

#l5="ro"
#l6="de"
#l7="fr"
#l8="fi"


for lang in ['spanish', 'polish', 'turkish', 'swedish']:
    if not os.path.exists(f'data/{lang}'):
        os.mkdir(f'data/{lang}')

langs = ['sv', 'tr', 'pl', 'es', 'ro', 'de', 'fr', 'fi']
#langs = ['fr', 'fi']

def subsample():

    if args.full:
        print("> Experiment mode, using full dataset but reduced queries..")
        save_path = "../../data"
    else:
        print("> Debug mode, subsamples data for algo dev..")
        save_path = "../../subsample"

    subdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_path)

    with open('./configs/data_paths.yml', 'r') as f:
        files = yaml.load(f)


    # load 
    en_docs = pd.read_csv(files['en_docs_org'], sep="\t", header=None, \
            names=['en_doc_id', 'title', 'content'])

    #en_l1_rel = pd.read_csv(files[f'en_{l1}_rel_org'], sep="\t", header=None, \
    #        names=['en_query_id', f'{l1}_doc_id', 'rel'])

    #en_l2_rel = pd.read_csv(files[f'en_{l2}_rel_org'], sep="\t", header=None, \
    #        names=['en_query_id', f'{l2}_doc_id', 'rel'])

    #en_l3_rel = pd.read_csv(files[f'en_{l3}_rel_org'], sep="\t", header=None, \
    #        names=['en_query_id', f'{l3}_doc_id', 'rel'])

    #en_l4_rel = pd.read_csv(files[f'en_{l4}_rel_org'], sep="\t", header=None, \
    #        names=['en_query_id', f'{l4}_doc_id', 'rel'])

    print("loading lang rels...")
    en_lang_rels = {}
    for l in langs:
        print(l, end=" ")
        en_lang_rels[l] = pd.read_csv(files[f'en_{l}_rel_org'], sep="\t", header=None, \
                names=['en_query_id', f'{l}_doc_id', 'rel'])


    en_queries = pd.read_csv(files['en_queries_org'], sep="\t", header=None, \
            names=['en_query_id', 'title', 'title_sentence'])

    en_query_doc_rel = pd.read_csv(files['en_query_doc_rel_org'], sep="\t", header=None, \
            names=['en_query_id', 'en_doc_id', 'rel'])

    print("loading raw..")
    lang_docs_raw = {}
    for l in langs:
        print(l, end=" ")
        lang_docs_raw[l] = pd.read_csv(files[f'{l}_docs_raw_org'], sep="\t",
        header=None, names=[f'{l}_doc_id', 'title', 'content'])

    #l1_docs_raw = pd.read_csv(files[f'{l1}_docs_raw_org'], sep="\t", header=None,\
    #        names=[f'{l1}_doc_id', 'title', 'content'])

    #ro_docs_tra = pd.read_csv(files['ro_docs_tra_org'], sep="\t", header=None, \
    #        names=['ro_doc_id', 'title', 'content'])

    #l2_docs_raw = pd.read_csv(files[f'{l2}_docs_raw_org'], sep="\t", header=None, \
    #        names=[f'{l2}_doc_id', 'title', 'content'])

    #l3_docs_raw = pd.read_csv(files[f'{l3}_docs_raw_org'], sep="\t", header=None, \
    #        names=[f'{l3}_doc_id', 'title', 'content'])

    #l4_docs_raw = pd.read_csv(files[f'{l4}_docs_raw_org'], sep="\t", header=None, \
    #        names=[f'{l4}_doc_id', 'title', 'content'])


    # subsample
    if args.full:
        pass
    else:
        #en_l1_rel = en_l1_rel[0:args.n]
        en_lang_rels[l] = en_lang_rels[l][0:args.n]
        
    ### ====================================================================
    ### Test and Valid sets (exact match and intersect acl1ss all languages)
    ### ===================================================================
    #en_l1_rel = en_l1_rel[en_l1_rel['rel']==2]
    #en_l2_rel = en_l2_rel[en_l2_rel['rel']==2]
    #en_l3_rel = en_l3_rel[en_l3_rel['rel']==2]
    #en_l4_rel = en_l4_rel[en_l4_rel['rel']==2]

    for l in langs:
        print(l, end=" ")
        en_lang_rels[l] = en_lang_rels[l][en_lang_rels[l]['rel']==2]

    en_en_rel = en_query_doc_rel[en_query_doc_rel['rel']==2]

    # ensure that all docs are in en_ro_rel, en_de_rel, en_en_rel
    en_doc_ids = en_docs['en_doc_id'].values
    en_query_ids = en_queries['en_query_id'].values
    en_query_ids2 = en_en_rel['en_query_id'].values
    en_query_ids = list(set(en_query_ids).intersection(set(en_query_ids2)))

    print("prep lang doc ids")
    lang_doc_ids = {}
    for l in langs:
        print(l, end=" ")
        lang_doc_ids[l] = set(lang_docs_raw[l][f'{l}_doc_id'].values)

    #l1_doc_ids = set(l1_docs_raw[f'{l1}_doc_id'].values)
    #l2_doc_ids = set(l2_docs_raw[f'{l2}_doc_id'].values)
    #l3_doc_ids = set(l3_docs_raw[f'{l3}_doc_id'].values)
    #l4_doc_ids = set(l4_docs_raw[f'{l4}_doc_id'].values)

    en_en_rel = en_en_rel[(en_en_rel['en_doc_id'].isin(en_doc_ids))\
            & (en_en_rel['en_query_id'].isin(en_query_ids))]
    en_query_ids = en_en_rel['en_query_id'].values

    print("prep lang rels intersect")
    for l in langs:
        print(l, end=" ")
        en_lang_rels[l] = en_lang_rels[l][(en_lang_rels[l][f'{l}_doc_id'].isin(lang_doc_ids[l]))
                & (en_lang_rels[l]['en_query_id'].isin(en_query_ids))]

    #en_l1_rel = en_l1_rel[(en_l1_rel[f'{l1}_doc_id'].isin(l1_doc_ids)) \
    #        & (en_l1_rel['en_query_id'].isin(en_query_ids))]

    #en_l2_rel = en_l2_rel[(en_l2_rel[f'{l2}_doc_id'].isin(l2_doc_ids)) \
    #        & (en_l2_rel['en_query_id'].isin(en_query_ids))]

    #en_l3_rel = en_l3_rel[(en_l3_rel[f'{l3}_doc_id'].isin(l3_doc_ids)) \
    #        & (en_l3_rel['en_query_id'].isin(en_query_ids))]

    #en_l4_rel = en_l4_rel[(en_l4_rel[f'{l4}_doc_id'].isin(l4_doc_ids)) \
    #        & (en_l4_rel['en_query_id'].isin(en_query_ids))]


    # do a big join over the en-sw-de-rel
    print("big join")
    big_join = en_en_rel.merge(en_lang_rels[langs[0]], how="inner", on="en_query_id", suffixes=("_en",
        f"_{langs[0]}"))

    for l in langs[1:]:
        print(l, end=" ")
        big_join = big_join.merge(en_lang_rels[l], how='inner', on='en_query_id')

    #big_join = big_join.merge(en_l2_rel, how="inner", on="en_query_id")
    #big_join = big_join.merge(en_l3_rel, how="inner", on="en_query_id", suffixes=("_de", "_fr"))
    #big_join = big_join.merge(en_l4_rel, how="inner", on="en_query_id")

    big_join['Q0'] = ['Q0' for i in range(len(big_join))]
    big_join['rel'] = [1 for i in range(len(big_join))]


    print("drop duplicates")

    for l in langs:
        print(l, end=" ")
        big_join.drop_duplicates(f'{l}_doc_id', inplace=True)
    

    #big_join.drop_duplicates(f'{l1}_doc_id', inplace=True)
    #big_join.drop_duplicates(f'{l2}_doc_id', inplace=True)
    big_join.drop_duplicates('en_doc_id', inplace=True)
    #big_join.drop_duplicates(f'{l3}_doc_id', inplace=True)
    #big_join.drop_duplicates(f'{l4}_doc_id', inplace=True)

    ##########################
    # Split Test validation 1000 docs each
    # 
    print("Subset test-valid documents", len(big_join))
    

    big_join_test_valid = big_join.sample(2000, replace=False, random_state=random_state)
    #big_join_test = big_join.sample(1000, replace=False, random_state=random_state)

    # used to filter out train docs
    test_valid_query_ids = big_join_test_valid['en_query_id'].values
    #test_query_ids = big_join_test['en_query_id'].values

    big_join_test = big_join_test_valid.sample(1000, replace=False, random_state=random_state)
    big_join_valid = big_join_test_valid.drop(big_join_test.index)
    big_join_train = big_join.drop(big_join_test_valid.index)

    print("subset docs")
    for l in langs:
        print(l, end=" ")
        subset_docs(l, lang_docs_raw[l], big_join_test, files, mode="test")
        subset_docs(l, lang_docs_raw[l], big_join_valid, files, mode="valid")

    # take care of en
    

    test_en_ids = big_join_test['en_doc_id'].values
    test_en_docs_ = en_docs[en_docs['en_doc_id'].isin(test_en_ids)]
    #test_en_docs_.to_csv(files['en_docs']+ f".test{grp}", sep="\t", header=None, index=False)
    test_en_docs_.to_csv(files['en_docs']+ ".test", sep="\t", header=None, index=False)

    valid_en_ids = big_join_valid['en_doc_id'].values
    valid_en_docs_ = en_docs[en_docs['en_doc_id'].isin(valid_en_ids)]
    #valid_en_docs_.to_csv(files['en_docs']+ f".valid{grp}", sep="\t", header=None, index=False)
    valid_en_docs_.to_csv(files['en_docs']+ f".valid", sep="\t", header=None, index=False)

 
    ##########################################################################
    # Train documents
    # Get all exact matches, and write to train file.
    # need to drop duplicates cos not one to one matching.. 
    # Earlier already made sure the doc exist in en_docs and language_docs
    for l in langs:
        save_train_docs(files, l, en_lang_rels[l], lang_docs_raw[l], en_en_rel, en_docs,
                test_valid_query_ids)
    
    #save_train_docs(files, f'{l1}', en_l1_rel, l1_docs_raw, en_en_rel, en_docs, test_query_ids)
    #save_train_docs(files, f'{l2}', en_l2_rel, l2_docs_raw, en_en_rel, en_docs, test_query_ids)
    #save_train_docs(files, f'{l3}', en_l3_rel, l3_docs_raw, en_en_rel, en_docs, test_query_ids)
    #save_train_docs(files, f'{l4}', en_l4_rel, l4_docs_raw, en_en_rel, en_docs, test_query_ids)

def subset_docs(lang, lang_docs_raw, big_join_mode, files, mode="test"):
    l_doc_id = '{}_doc_id'.format(lang)

    test_ids = big_join_mode[l_doc_id].values
    en_lang_rel_gold = big_join_mode[['en_doc_id', 'Q0', l_doc_id, 'rel_en']]
    lang_docs_raw = lang_docs_raw[lang_docs_raw[l_doc_id].isin(test_ids)]

    rel_saveto = files['en_{}_rel'.format(lang)] + "."+mode
    docs_saveto = files['parallel_{}'.format(lang)] + "."+mode

    en_lang_rel_gold.to_csv(rel_saveto, sep="\t", header=None, index=False)
    lang_docs_raw.to_csv(docs_saveto, sep="\t", header=None, index=False)
    print(" Save wiki_{}.documents {}:{}".format(lang, mode, len(lang_docs_raw)))


# deprecated
def save_train_docs(files, lang, en_l_rel, l_docs_raw, en_en_rel, en_docs, test_query_ids):

    en_l_rel_train = en_l_rel[~en_l_rel['en_query_id'].isin(test_query_ids)]
    en_l_rel_train.drop_duplicates(['{}_doc_id'.format(lang)], inplace=True)
    en_l_rel_train.drop_duplicates(['en_query_id'], inplace=True)

    if len(en_l_rel_train)>8000:
        print("reducing {} docs from {} to 8000".format(lang, len(en_l_rel_train)))
        en_l_rel_train = en_l_rel_train.sample(n=8000, random_state=random_state)

    # subsample first so that it is easier to find later
    l_docs_raw = l_docs_raw[l_docs_raw['{}_doc_id'.format(lang)].isin(en_l_rel_train['{}_doc_id'.format(lang)])]
    en_en_rel = en_en_rel[en_en_rel['en_query_id'].isin(en_l_rel_train['en_query_id'])]
    en_docs = en_docs[en_docs['en_doc_id'].isin(en_en_rel['en_doc_id'])]
    
    l_docs_train = []
    enl_docs_train = []
    for i in range(len(en_l_rel_train)):
        query_id = en_l_rel_train.iloc[i].en_query_id
        en_doc_id = en_en_rel[en_en_rel['en_query_id']==query_id].en_doc_id.values[0]
        l_doc_id = en_l_rel_train.iloc[i]['{}_doc_id'.format(lang)]

        try:
            enl_docs_train.append(en_docs[en_docs['en_doc_id']==en_doc_id].values[0])
            l_docs_train.append(l_docs_raw[l_docs_raw['{}_doc_id'.format(lang)]==l_doc_id].values[0])
        except:
            print("Not found:", en_doc_id, l_doc_id)
    assert len(l_docs_train)==len(enl_docs_train), pdb.set_trace()
    l_docs_train = pd.DataFrame(l_docs_train)
    enl_docs_train = pd.DataFrame(enl_docs_train)
    print(" Save wiki_{}.documents.train:{}".format(lang, len(l_docs_train)))

    l_docs_train.to_csv(files['parallel_{}'.format(lang)]+".train", sep="\t", header=None, index=False)
    enl_docs_train.to_csv(files['parallel_en{}'.format(lang)]+".train", sep="\t", header=None, index=False)
#


if __name__=="__main__":
    subsample()
