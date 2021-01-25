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

    en_ro_rel = pd.read_csv(files['en_ro_rel_org'], sep="\t", header=None, \
            names=['en_query_id', 'ro_doc_id', 'rel'])

    en_de_rel = pd.read_csv(files['en_de_rel_org'], sep="\t", header=None, \
            names=['en_query_id', 'de_doc_id', 'rel'])

    en_fr_rel = pd.read_csv(files['en_fr_rel_org'], sep="\t", header=None, \
            names=['en_query_id', 'fr_doc_id', 'rel'])

    en_fi_rel = pd.read_csv(files['en_fi_rel_org'], sep="\t", header=None, \
            names=['en_query_id', 'fi_doc_id', 'rel'])


    en_queries = pd.read_csv(files['en_queries_org'], sep="\t", header=None, \
            names=['en_query_id', 'title', 'title_sentence'])

    en_query_doc_rel = pd.read_csv(files['en_query_doc_rel_org'], sep="\t", header=None, \
            names=['en_query_id', 'en_doc_id', 'rel'])

    ro_docs_raw = pd.read_csv(files['ro_docs_raw_org'], sep="\t", header=None,\
            names=['ro_doc_id', 'title', 'content'])

    #ro_docs_tra = pd.read_csv(files['ro_docs_tra_org'], sep="\t", header=None, \
    #        names=['ro_doc_id', 'title', 'content'])

    de_docs_raw = pd.read_csv(files['de_docs_raw_org'], sep="\t", header=None, \
            names=['de_doc_id', 'title', 'content'])

    fr_docs_raw = pd.read_csv(files['fr_docs_raw_org'], sep="\t", header=None, \
            names=['fr_doc_id', 'title', 'content'])

    fi_docs_raw = pd.read_csv(files['fi_docs_raw_org'], sep="\t", header=None, \
            names=['fi_doc_id', 'title', 'content'])


    # subsample
    if args.full:
        pass
    else:
        en_ro_rel = en_ro_rel[0:args.n]
        
    ### ====================================================================
    ### Test and Valid sets (exact match and intersect across all languages)
    ### ===================================================================
    en_ro_rel = en_ro_rel[en_ro_rel['rel']==2]
    en_de_rel = en_de_rel[en_de_rel['rel']==2]
    en_fr_rel = en_fr_rel[en_fr_rel['rel']==2]
    en_fi_rel = en_fi_rel[en_fi_rel['rel']==2]
    en_en_rel = en_query_doc_rel[en_query_doc_rel['rel']==2]

    # ensure that all docs are in en_ro_rel, en_de_rel, en_en_rel
    en_doc_ids = en_docs['en_doc_id'].values
    en_query_ids = en_queries['en_query_id'].values
    en_query_ids2 = en_en_rel['en_query_id'].values
    en_query_ids = list(set(en_query_ids).intersection(set(en_query_ids2)))

    ro_doc_ids = set(ro_docs_raw['ro_doc_id'].values)
    de_doc_ids = set(de_docs_raw['de_doc_id'].values)
    fr_doc_ids = set(fr_docs_raw['fr_doc_id'].values)
    fi_doc_ids = set(fi_docs_raw['fi_doc_id'].values)

    en_en_rel = en_en_rel[(en_en_rel['en_doc_id'].isin(en_doc_ids))\
            & (en_en_rel['en_query_id'].isin(en_query_ids))]
    en_query_ids = en_en_rel['en_query_id'].values

    en_ro_rel = en_ro_rel[(en_ro_rel['ro_doc_id'].isin(ro_doc_ids)) \
            & (en_ro_rel['en_query_id'].isin(en_query_ids))]

    en_de_rel = en_de_rel[(en_de_rel['de_doc_id'].isin(de_doc_ids)) \
            & (en_de_rel['en_query_id'].isin(en_query_ids))]

    en_fr_rel = en_fr_rel[(en_fr_rel['fr_doc_id'].isin(fr_doc_ids)) \
            & (en_fr_rel['en_query_id'].isin(en_query_ids))]

    en_fi_rel = en_fi_rel[(en_fi_rel['fi_doc_id'].isin(fi_doc_ids)) \
            & (en_fi_rel['en_query_id'].isin(en_query_ids))]



    # do a big join over the en-sw-de-rel
    big_join = en_en_rel.merge(en_ro_rel, how="inner", on="en_query_id", suffixes=("_en",
        "_ro"))
    big_join = big_join.merge(en_de_rel, how="inner", on="en_query_id")
    big_join = big_join.merge(en_fr_rel, how="inner", on="en_query_id", suffixes=("_de", "_fr"))
    big_join = big_join.merge(en_fi_rel, how="inner", on="en_query_id")

    big_join['Q0'] = ['Q0' for i in range(len(big_join))]
    big_join['rel'] = [1 for i in range(len(big_join))]

    big_join.drop_duplicates('ro_doc_id', inplace=True)
    big_join.drop_duplicates('de_doc_id', inplace=True)
    big_join.drop_duplicates('en_doc_id', inplace=True)
    big_join.drop_duplicates('fr_doc_id', inplace=True)
    big_join.drop_duplicates('fi_doc_id', inplace=True)

    if len(big_join)>1000:
        print("Subset 1000 test documents")
        big_join = big_join.sample(1000, replace=False, random_state=42)


    test_ro_ids = big_join['ro_doc_id'].values
    test_de_ids = big_join['de_doc_id'].values
    test_en_ids = big_join['en_doc_id'].values
    test_fr_ids = big_join['fr_doc_id'].values
    test_fi_ids = big_join['fi_doc_id'].values
    test_query_ids = big_join['en_query_id'].values
    
    en_ro_rel_gold = big_join[['en_query_id', 'Q0', 'ro_doc_id', "rel_en"]]
    en_de_rel_gold = big_join[['en_query_id', 'Q0', 'de_doc_id', "rel_en"]]
    en_fr_rel_gold = big_join[['en_query_id', 'Q0', 'fr_doc_id', "rel_en"]]
    en_fi_rel_gold = big_join[['en_query_id', 'Q0', 'fi_doc_id', "rel_en"]]

    ro_docs_raw_ = ro_docs_raw[ro_docs_raw['ro_doc_id'].isin(test_ro_ids)]
    #ro_docs_tra_ = ro_docs_tra[ro_docs_tra['ro_doc_id'].isin(test_ro_ids)]
    de_docs_raw_ = de_docs_raw[de_docs_raw['de_doc_id'].isin(test_de_ids)]
    fr_docs_raw_ = fr_docs_raw[fr_docs_raw['fr_doc_id'].isin(test_fr_ids)]
    fi_docs_raw_ = fi_docs_raw[fi_docs_raw['fi_doc_id'].isin(test_fi_ids)]

    en_queries_ = en_queries[en_queries['en_query_id'].isin(test_query_ids)]

    assert (len(ro_docs_raw_)==len(de_docs_raw_)==len(en_queries_)==len(fr_docs_raw_)==len(fi_docs_raw_)), pdb.set_trace()
    
    # dump data
    print(" Intersection:", len(test_en_ids))
    print(" Save ens2sw.rel", len(en_ro_rel_gold))
    print(" Save ens2de.rel", len(en_de_rel_gold))

    en_ro_rel_gold.to_csv(files['en_ro_rel']+".test", sep="\t", header=None,index=False)
    en_de_rel_gold.to_csv(files['en_de_rel']+".test", sep="\t", header=None,index=False)
    en_fr_rel_gold.to_csv(files['en_fr_rel']+".test", sep="\t", header=None,index=False)
    en_fi_rel_gold.to_csv(files['en_fi_rel']+".test", sep="\t", header=None,index=False)

    print(" Save test wiki_en.queries", len(en_queries_))
    en_queries_.to_csv(files['en_queries']+".test", sep="\t", header=None, index=False)    
    
    #print(" Save wiki_simple.documents", len(en_docs_))
    en_docs_.to_csv(files['en_docs']+".test", sep="\t", header=None, index=False)

   # en_query_doc_rel_.to_csv(files['en_query_doc_rel']+".test", sep="\t", header=None, index=False)

    print(" Save wiki_ro.documents", len(ro_docs_raw_))
    #print(" Save wiki_ro.documents.translated", len(ro_docs_tra_))

    ro_docs_raw_.to_csv(files['parallel_ro']+".test", sep="\t", header=None, index=False)
    #ro_docs_tra_.to_csv(files['ro_docs_tra']+".test", sep="\t", header=None, index=False)

    print(" Save wiki_de.documents", len(de_docs_raw_))
    de_docs_raw_.to_csv(files['parallel_de']+".test", sep="\t", header=None, index=False)

    print(" Save wiki_fr.documents", len(fr_docs_raw_))
    fr_docs_raw_.to_csv(files['parallel_fr']+".test", sep="\t", header=None, index=False)

    print(" Save wiki_fi.documents", len(fi_docs_raw_))
    fi_docs_raw_.to_csv(files['parallel_fi']+".test", sep="\t", header=None, index=False)
 
    ##########################################################################
    # Train documents
    # Get all exact matches, and write to train file.
    # need to drop duplicates cos not one to one matching.. 
    # Earlier already made sure the doc exist in en_docs and language_docs
    
    save_train_docs(files, "ro", en_ro_rel, ro_docs_raw, en_en_rel, en_docs, test_query_ids)
    save_train_docs(files, "de", en_de_rel, de_docs_raw, en_en_rel, en_docs, test_query_ids)
    save_train_docs(files, "fr", en_fr_rel, fr_docs_raw, en_en_rel, en_docs, test_query_ids)
    save_train_docs(files, "fi", en_fi_rel, fi_docs_raw, en_en_rel, en_docs, test_query_ids)

def save_train_docs(files, lang, en_l_rel, l_docs_raw, en_en_rel, en_docs, test_query_ids):
    en_l_rel_train = en_l_rel[~en_l_rel['en_query_id'].isin(test_query_ids)]
    en_l_rel_train.drop_duplicates(['{}_doc_id'.format(lang)], inplace=True)
    en_l_rel_train.drop_duplicates(['en_query_id'], inplace=True)

    if len(en_l_rel_train)>10000:
        print("reducing {} docs from {} to 8000".format(lang, len(en_l_rel_train)))
        en_l_rel_train = en_l_rel_train.sample(n=8000, random_state=42)

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

    l_docs_train.to_csv(files['parallel_{}'.format(lang)]+".train", sep="\t", header=None,\
    index=False)
    enl_docs_train.to_csv(files['parallel_en{}'.format(lang)]+".train", sep="\t", header=None, index=False)




    # Do swahili
    #en_sw_rel_train = en_sw_rel[~en_sw_rel['en_query_id'].isin(test_query_ids)]

    #sw_docs_train = []
    #ensw_docs_train = []
    #for i in range(len(en_sw_rel_train)):
    #    query_id = en_sw_rel_train.iloc[i].en_query_id
    ##    en_doc_id = en_en_rel[en_en_rel['en_query_id']==query_id].en_doc_id.values[0]
    #    sw_doc_id = en_sw_rel_train.iloc[i].sw_doc_id
            
    #    try:
    #        ensw_docs_train.append(en_docs[en_docs['en_doc_id']==en_doc_id].values[0])
    #        sw_docs_train.append(sw_docs_raw[sw_docs_raw['sw_doc_id']==sw_doc_id].values[0])
    #    except:
    #        print("not found:", en_doc_id, sw_doc_id)

    #assert len(sw_docs_train) == len(ensw_docs_train), pdb.set_trace()
    #sw_docs_train = pd.DataFrame(sw_docs_train)
    #ensw_docs_train = pd.DataFrame(ensw_docs_train)

    #print(" Save wiki_sw.documents.train", len(sw_docs_train))
    #sw_docs_train.to_csv(files['parallel_sw']+".train", sep="\t", header=None, index=False)
    #ensw_docs_train.to_csv(files['parallel_ensw']+".train", sep="\t", header=None, index=False)

    # Do German
    #en_de_rel_train = en_de_rel[~en_de_rel['en_query_id'].isin(test_query_ids)]
    #en_de_rel_train.drop_duplicates(['de_doc_id'], inplace=True)
    #en_de_rel_train.drop_duplicates(['en_query_id'], inplace=True)

    #if len(en_de_rel_train)>10000:
    #    print(len(en_de_rel_train))
    #    en_de_rel_train = en_de_rel_train.sample(n=10000, random_state=1)

    # subsample first so that it is easier to find later
    #de_docs_raw = de_docs_raw[de_docs_raw['de_doc_id'].isin(en_de_rel_train['de_doc_id'])]
    #en_en_rel = en_en_rel[en_en_rel['en_query_id'].isin(en_de_rel_train['en_query_id'])]
    #en_docs = en_docs[en_docs['en_doc_id'].isin(en_en_rel['en_doc_id'])]
    
    ##de_docs_train = []
    #ende_docs_train  = []
    #for i in range(len(en_de_rel_train)):
        #if i%1000==0:
   #     query_id = en_de_rel_train.iloc[i].en_query_id
   #     en_doc_id = en_en_rel[en_en_rel['en_query_id']==query_id].en_doc_id.values[0]
   #     de_doc_id = en_de_rel_train.iloc[i].de_doc_id
   #     try:
   #         ende_docs_train.append(en_docs[en_docs['en_doc_id']==en_doc_id].values[0])
   #         de_docs_train.append(de_docs_raw[de_docs_raw['de_doc_id']==de_doc_id].values[0])
   #     except:
   #         print("not found:", en_doc_id, de_doc_id)

   # assert len(de_docs_train) == len(ende_docs_train), pdb.set_trace()
   # de_docs_train = pd.DataFrame(de_docs_train)
   # ende_docs_train = pd.DataFrame(ende_docs_train)

    #print(" Save wiki_de.documents.train", len(de_docs_train))
    #de_docs_train.to_csv(files['parallel_de']+".train", sep="\t", header=None, index=False)
    #ende_docs_train.to_csv(files['parallel_ende']+".train", sep="\t", header=None, index=False)


if __name__=="__main__":
    subsample()
