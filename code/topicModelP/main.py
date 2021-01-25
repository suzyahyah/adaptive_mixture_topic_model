#!/usr/bin/python/3.6
# Author: Suzanna Sia
import sys
import os
import numpy as np
import utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pstats, cProfile
#from language import Language

import yaml
import pandas as pd 
import time
from math import exp

from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit as sss
from sklearn.model_selection import ShuffleSplit 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', dest="data", default="dev", help="dev, sub, full gen")
parser.add_argument('--model', dest="model", default="vanilla", help="vanilla or gauss")
parser.add_argument("--niter", dest="niter", default=10, type=int)
parser.add_argument("--ndim", dest="ndim", default=300, type=int)
parser.add_argument("--ntopics", dest="ntopics", default=25, type=int)
parser.add_argument("--cython", dest="c", type=int, help="use cython")
parser.add_argument("--kmeans", dest="use_kmeans", type=str, help="use cython")
parser.add_argument("--num_lang", dest="num_lang", type=int, help="use cython")
parser.add_argument("--shared_params", dest="shared_params", type=int, help="use cython")
parser.add_argument("--sample_mean", dest="sample_mean", type=int, help="use cython")
parser.add_argument("--temp", dest="temp", type=int, default=1)
parser.add_argument("--interpolate", dest="interpolate", default=1.0, type=float)
parser.add_argument("--dev", dest="dev", default=1.0, type=float)
parser.add_argument("--stagger", dest="stagger", default=0, type=int)

parser.add_argument("--train_size", dest="train_size", default=1000, type=int)
parser.add_argument("--split", dest="split", default=0, type=int)

parser.add_argument("--data_path", type=str, default="configs/data_paths.yml")
parser.add_argument("--trecin_dir", dest="trecin_dir", type=str)
parser.add_argument("--trecout_dir", dest="trecout_dir", type=str)
parser.add_argument("--topic_prop_dir", dest="topic_prop_dir", type=str)
parser.add_argument("--llh_dir", dest="llh_dir", type=str)
parser.add_argument("--hgs_dir", dest="hgs_dir", type=str)
parser.add_argument("--pickle_dir", dest="pickle_dir", type=str)
parser.add_argument("--wordsout_dir", dest="wordsout_dir", type=str)
parser.add_argument("--train_prop_dir", dest="train_prop_dir", type=str)
parser.add_argument("--entropy_dir", dest="entropy_dir", type=str)
parser.add_argument("--load_iter", dest="load_iter", type=int)
parser.add_argument("--train_test_valid", dest="train_test_valid", type=str)
parser.add_argument("--short_docs", dest="short_docs", type=int)

parser.add_argument("--cov0_scalar", dest="cov0_scalar", type=float)
parser.add_argument("--mu0_b", dest="mu0_b", type=float)
parser.add_argument("--max_eval", dest="max_eval", type=int)
parser.add_argument("--scaling_dof0", dest="scaling_dof0", type=int)
parser.add_argument("--beta_beta", type=int, default=10)

# only for simulation
parser.add_argument("--ndocs_per_topic", dest="ndocs", type=int)

args = parser.parse_args()

#### IMPORT C files or python files
from language_c2 import Language
from ml_gaussLDA import ml_gaussLDA

def main():

    with open("./configs/data_paths.yml", 'r') as f:
        files = yaml.load(f)

    global pickle_dir
    pickle_dir = os.path.join(os.getcwd(), args.pickle_dir)

    ### NEWS
    if "ne-" in args.data:

        mode = args.data[args.data.find('-')+1:]

        if mode=="all":
            train_data = fetch_20newsgroups(data_home="./data", subset="train", \
                    remove=('headers', 'footers', 'quotes'))
            test_data = fetch_20newsgroups(data_home="./data", subset="test", \
                    remove=('headers', 'footers', 'quotes'))

            train_data = [d.replace('\n', ' ') for d in train_data['data']]
            test_data = [d.replace('\n', ' ') for d in test_data['data']]

            embed_path = files['en_multi_embed']
            tr_lang = Language("en", args.ndim, train_data, embed_path, args.ntopics)
            te_lang = Language("en", args.ndim, test_data, embed_path, args.ntopics)

            
            tr_lang.load_embeddings(filter=True)

            temp_fn = "vocab_temp/vocab-{}-train{}-intp{}.{}.{}".format(str(args.ntopics),
                args.train_size, args.interpolate, args.split, "en")
            tr_lang.write_vocab(temp_fn)
            te_lang = Language("en", args.ndim, test_data, embed_path, args.ntopics, temp_fn)



            languages = [tr_lang]
            te_lang = [te_lang]
        else:
            train_data = fetch_20newsgroups(data_home="./data", subset="all",\
                    remove=('headers', 'footers', 'quotes'))

            targets = train_data['target']
            train_data = [d.replace('\n', ' ') for d in train_data['data']]
            #train_data = train_data[:50]

            print("Loading 20 ng:{}, length of data:{}".format(mode, len(train_data)))
            #refactor
            embed_path = files['en_multi_embed']

            # stratified shuffle split
            ssplit = sss(n_splits=7, test_size=8000, train_size=args.train_size, random_state=0)
            train_ix, test_ix = list(ssplit.split(train_data, targets))[int(args.split)]

            tr_lang, te_lang = process_shufflesplit("en", embed_path, train_ix, test_ix, train_data)
            languages = [tr_lang]
            te_lang = [te_lang]

    elif args.data.startswith('en'):

        l1 = "en"
        l2 = args.data[-2:]

        file1 = files['parallel_en{}'.format(l2)]
        file2 = files['parallel_{}'.format(l2)]
        test1 = files['en_docs']
        embed1 = files['en_multi_embed']
        embed2 = files['{}_multi_embed'.format(l2)]

        # these are nicely ordered.
        _, train_data1 = utils.read_data(file1 + ".train")
        _, train_data2 = utils.read_data(file2 + ".train")

        ss = ShuffleSplit(n_splits=10, train_size=args.train_size, test_size=1000, random_state=0)
        train_ix, test_ix = list(ss.split(train_data1))[int(args.split)]

        tr1, te1 = process_shufflesplit(l1, embed1, train_ix, test_ix, train_data1)
        tr2, te2 = process_shufflesplit(l2, embed2, train_ix, test_ix, train_data2)

        languages = [tr1, tr2]
        te_lang = [te1, te2]

        print(f"TRAIN1:{tr1.ndocs}, TRAIN2:{tr2.ndocs}, TEST1:{te1.ndocs}, TEST2:{te2.ndocs}")

    sys.stdout.flush()

    if args.model == "gmm":
        print("\n===> Running gaussian mixture model:")

        gmm = GMM(languages, args.ntopics, args.ndim)
        gmm.initialise(l=0)
        gmm.run_model(l=0)
        total = 0
        proportions = []
        for k in range(args.ntopics):
            topic_count = len(gmm.cluster_dict[k])
            total += topic_count
            proportions.append(topic_count)

        proportions = [count/total for count in proportions]

        for k in range(args.ntopics):
            gmm.cluster_dict[k].sort(key=lambda x: x[1])
            words = gmm.cluster_dict[k][-15:]
            words = [w[0] for w in words]
            print(k, ";", proportions[k], ";", " ".join(words))
        
        word_file = os.path.join(args.wordsout_dir, languages[0].name+"_words.txt")
        sys.exit(0)

    model = model_gauss(languages)
    starttime = time.time()

    if args.train_test_valid == "external_score":
        for score_type in ["cv", "npmi"]:
            utils.external_score_top_words(args.wordsout_dir, score_type=score_type)
        sys.exit(0)


    elif args.train_test_valid=="top_words":
        assert args.load_iter>0, "args.load_iter must be > 0"
        suffix="niter{}.n{}.s{}".format(str(args.load_iter), str(languages[0].ndocs), str(args.split))
            #model.load(pickle_dir, suffix="niter{}".format(str(args.load_iter)))
        model.load(pickle_dir, suffix=suffix)
        print("loaded:", suffix)
        wordsout_dir = os.path.join(args.wordsout_dir, f"{args.train_size}.split{args.split}")
        if not os.path.isdir(wordsout_dir):
            os.makedirs(wordsout_dir)

        print("write top words to:", wordsout_dir)
        model.write_top_words(wordsout_dir)


#        model.print_entropy_stats(args.entropy_dir)

    elif args.train_test_valid=="study_beta":
        model.initialise()
        fn = f'news_logs/beta_values4_{args.data}.txt'
        model.run_gibbs_sampling(1, args.load_iter, te_lang=te_lang)
        model.calc_beta_values(fn)
        sys.exit("fin")

    elif args.train_test_valid=="train":
        if args.data.startswith('ne'):
            savef = 'news_logs/all_results.txt'
        elif args.data.startswith('en'):
            savef = f'wiki_logs/{l2}-all_results.txt'
        else:
            sys.exit("invalid option for args.data")

        if args.load_iter==0:
            model.initialise()
            model.run_gibbs_sampling(args.niter, args.load_iter, te_lang=te_lang)
            print("write top words to:", args.wordsout_dir)

            model.write_top_words(args.wordsout_dir)
            #model.print_entropy_stats(args.entropy_dir)
           
        else:
            pf = os.listdir(pickle_dir)
            ndocs = languages[0].ndocs
            ldkfiles = [f for f in pf if f.endswith(f'n{ndocs}.s{args.split}_l_d_k')]
            ldkfiles = [f[5:f.find('.')] for f in ldkfiles]
            
            if len(ldkfiles)==0:
                print("failed to load, nothing found.. Rerun gibbs sampling")
                load_iter=0
                model.initialise()
                model.run_gibbs_sampling(args.niter, load_iter, te_lang=te_lang)
            else:
                load_iter = max([int(i) for i in ldkfiles])
                suffix="niter{}.n{}.s{}".format(str(load_iter), str(ndocs), str(args.split))
                print("trying to load:", suffix, load_iter)

                if f'Posterior{suffix}0_upperT_allk' not in pf:
                    #print("os remove old files")
                    #os.remove(os.path.join(pickle_dir, f"{suffix}_l_d_k"))
                    sys.exit("Error: uperT_allk not found")

                print("Loading max:", load_iter)
                model.load(pickle_dir, suffix)
                model.run_gibbs_sampling(args.niter-load_iter+2, loaditer=(load_iter+1), te_lang=te_lang)
        with open(savef, 'a') as f:
            if args.interpolate!=0.5:
                intp = int(args.interpolate)
            else:
                intp = args.interpolate
            f.write(f'{args.shared_params}\t \
                    {intp}\t \
                    {args.train_size}\t \
                    {args.split}\t{model.get_test_coh()}\
                    \t{model.get_train_coh()}\n')\


    else:
        ir_test(model, languages, mode=args.train_test_valid)

    print("Time taken:", time.time()-starttime)

def model_gauss(languages, assign_topics=[]):

    model = ml_gaussLDA(languages, args.ntopics, args.ndim, args.niter, args.ntopics,
            0.01, args.pickle_dir, args.train_prop_dir,
            args.llh_dir, args.hgs_dir, topic_assignment=assign_topics,
            interpolate=args.interpolate, shared_params=args.shared_params,
            prior_mean=args.sample_mean, stagger=args.stagger, mu0_b=args.mu0_b,
            cov0_scalar=args.cov0_scalar, max_eval=args.max_eval,
            scaling_dof0=args.scaling_dof0, 
            beta_beta=args.beta_beta, seed=args.split)

    return model

def ir_test(model, languages, mode=args.train_test_valid):

    te_ix1, test_1, te_ix2, test_2 = utils.getData(args.data_path, args.data, mode=mode)

    print("--> LOAD MODEL: Testing from model@iteration:{}, run inference for {} gibbs \
            iterations".format(args.load_iter, args.niter))
 
    model.load(args.pickle_dir, suffix="niter{}".format(str(args.load_iter)))
    all_kprobs_2, invalid_doc_ix_2 = model.infer_test_distributions(test_2, iters=args.niter, l=1)

    print("--> RELOAD MODEL")
    model = model_gauss(languages)
    model.load(args.pickle_dir, suffix="niter{}".format(str(args.load_iter)))
    all_kprobs_1, invalid_doc_ix_1 = model.infer_test_distributions(test_1, iters=args.niter, l=0)
    save_kprob(all_kprobs_1, languages[0].name, mode)
    save_kprob(all_kprobs_2, languages[1].name, mode)
    sim_scores = cosine_similarity(all_kprobs_1, all_kprobs_2)

    # generate ir files
    assert sim_scores.shape[0] == sim_scores.shape[1]
    to_write = []
    for i in range(sim_scores.shape[0]):
        ranking = np.argsort(sim_scores[i,:])
        for j in range(sim_scores.shape[1]):
            rank = ranking[j]
            score = sim_scores[i, j]
            result = f"{te_ix1[i]}\tQ0\t{te_ix2[j]}\t{rank}\t{score}\tSTANDARD"
            to_write.append(result)

    save_path = os.path.join(args.trecin_dir, \
            f'trainiter{str(args.load_iter)}-testiter{str(args.niter)}.txt.{mode}') 
 
    print("SAVE PATH:", save_path)
    with open(save_path, 'w') as f:
        to_write = "\n".join(to_write)
        f.write(to_write)

def save_kprob(vals, name, mode):
    save_path = os.path.join(args.topic_prop_dir, f'{name}_{args.load_iter}.txt.{mode}')
    np.savetxt(save_path, vals, fmt='%.3f')

def process_shufflesplit(ell, embed_path, train_ix, test_ix, train_data):
    print("Reading split:", ell, args.split, len(train_ix), len(test_ix))
    tr = [train_data[ix] for ix in train_ix]
    te = [train_data[ix] for ix in test_ix]
    tr_lang = Language(ell, args.ndim, tr, embed_path, args.ntopics)
        
    if not os.path.isdir('vocab_temp'):
        os.mkdir('vocab_temp')

    temp_fn = "vocab_temp/vocab-{}-train{}-intp{}.{}.{}".format(str(args.ntopics),
        args.train_size, args.interpolate, args.split, ell)

    tr_lang.write_vocab(temp_fn)
    tr_lang.load_embeddings(filter=True)

    te_lang = Language(ell, args.ndim, te, embed_path, args.ntopics, temp_fn)

    return tr_lang, te_lang


if __name__ == "__main__":
    main()
