#!/usr/bin/python
# Author: Suzanna Sia
import numpy as np
import time
import os
import sys
import requests
from sklearn.metrics import v_measure_score
import itertools
import pandas as pd


def topic_coherence2(nl, langs, l_word_topic_counts, nwords, ntopics, ndocs, mode="npmi"):
    # from mimno paper
    # l_word_topic_counts must be from the topic model
    # count_matrix and word doc_counts should be from the test set
    l_topic_co_scores = np.zeros((nl, ntopics))
    for l in range(nl):
        # from base corpus
        count_matrix = np.asarray(langs[l].count_matrix)
        word_doc_counts = np.asarray(langs[l].word_doc_counts)
        
        # from model
        word_topic_counts = l_word_topic_counts[l]
        all_topics = []

        for k in range(ntopics):
            ixs = np.argsort(word_topic_counts[:, k])[::-1][:nwords]
        #    sorted_ixs = sorted(ixs)[::-1]
            topic_npmi = 0
            sorted_ixs = ixs

            all_pairs = list(itertools.combinations(sorted_ixs, 2))
            total_pairs = len(all_pairs)

            for p0, p1 in all_pairs:
                if p0 < p1:
                    i = p0
                    j = p1
                elif p0 > p1:
                    i = p1
                    j = p0

                if count_matrix[i,j]==0:
                    topic_npmi += -1

                else:
                    pmi = np.log((count_matrix[i, j])*ndocs / (word_doc_counts[i]*word_doc_counts[j]))
                    normalizer = -np.log((count_matrix[i, j])/ndocs)
                    topic_npmi += pmi/normalizer

            topic_npmi = topic_npmi/total_pairs
            l_topic_co_scores[l][k] = np.round(topic_npmi, 3)

    return l_topic_co_scores #, l_topic_co_scores2




def topic_coherence(nl, langs, l_word_topic_counts, nwords, ntopics, ndocs, mode="npmi"):
    # from mimno paper
    # l_word_topic_counts must be from the topic model
    # count_matrix and word doc_counts should be from the test set
    l_topic_co_scores = np.zeros((nl, ntopics))
    eps = 10**(-12)

    for l in range(nl):
        # from base corpus
        count_matrix = np.asarray(langs[l].count_matrix)
        word_doc_counts = np.asarray(langs[l].word_doc_counts)
        
        # from model
        word_topic_counts = l_word_topic_counts[l]

        for k in range(ntopics):
            ixs = np.argsort(word_topic_counts[:, k])[::-1][:nwords]
        #    sorted_ixs = sorted(ixs)[::-1]
            score1 = 0
            sorted_ixs = ixs

            if mode=="npmi":
                all_pairs = list(itertools.combinations(sorted_ixs, 2))
                total_pairs = len(all_pairs)

                for p0, p1 in all_pairs:
                    if p0 > p1:
                        i = p1
                        j = p0
                    else:
                        i = p0
                        j = p1

                # if the words do not exist in the base corpus, then we cannot evaluate this pair
                # if the words exist in the base corpus but the word pair does not, then the NPMI score should be -1. 
                    if word_doc_counts[i]==0 or word_doc_counts[j]==0:
                        total_pairs -=1
                    elif count_matrix[i,j]==0:
                        score1 -= 1
                    else:
                        #pmi = np.log((count_matrix[i, j])*ndocs
                        #/ (word_doc_counts[i]*word_doc_counts[j]))
                        pmi_w1w2 = np.log((w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps)
                        normalizer = -np.log(count_matrix[i, j]/ndocs + eps)
                        score1 += pmi/normalizer
            else:
                # mimno paper
                for i in range(1, len(sorted_ixs)):
                    for j in range(i):
                        ix1 = sorted_ixs[i]
                        ix2 = sorted_ixs[j]

                        if ix1<ix2:
                            score+=np.log((count_matrix[ix1, ix2] + 1)\
                                / (word_doc_counts[ix2]+1))
                        else:
                            score+= np.log((count_matrix[ix2, ix1]+1)/(word_doc_counts[ix2]+1))

            if total_pairs==0:
                score1 = -1
            else:
                score1 = score1/total_pairs
            l_topic_co_scores[l][k] = np.round(score1, 3)

    return l_topic_co_scores #, l_topic_co_scores2


def print_top_words(num_lang, langs, l_word_topic_counts, nWords, ntopics, betab_allk, data="",
        savepath="", l_topic_scores=None):

    for l in reversed(range(num_lang)):
        # reverse printing order for sanity
        if langs[l].embedding is None:
            has_embedding = False
        else:
            has_embedding = True

        print("language:", l)
        avg_purity = []
        top_topic_words = []

        for k in range(ntopics):
             
            largest_membership = np.argsort(l_word_topic_counts[l, :, k])[::-1][:nWords]
            try:
                words = [langs[l].ix2w[ix] for ix in largest_membership]
            except:
                words = [""]
                #print("Topic:{} insufficient words".format(k))
            top_words = []
            for w in words:
                if has_embedding:
                    if w not in langs[l].embedding.vocab:
                        w = w+"(oov)"

                top_words.append(w)
            
            if langs[0].name=="a":
                topics = np.zeros(ntopics)
                for w in words:
                    for kt in range(ntopics):
                        if int(w[0]) == kt:
                            topics[kt] +=1

                purity = topics[np.argmax(topics)]/topics.sum()
                avg_purity.append(purity)
                print("Topic:{} ({:.3f}) {}".format(np.argmax(topics), purity, \
                        " ".join(top_words).encode('utf-8')))
            else:
                #print("")
                if l_topic_scores is not None:
                    print("Topic:{} {} betab:{} cohsc:{} wc:{}".format(k, " ".join(top_words).encode('utf-8'), \
                        np.round(betab_allk[k],3),
                        l_topic_scores[l, k],
                        np.sum(l_word_topic_counts[l,:,k])))
                else:
                    pass

            top_words = " ".join(top_words)
            topic_counts = np.sum(l_word_topic_counts[l,:,k])
            total_counts = np.sum(l_word_topic_counts[l,:,:])
            prop = np.around(topic_counts/total_counts, 3)

            top_topic_words.append(str(k)+"\t"+str(prop)+"\t"+top_words)

        if l_topic_scores is not None:
            print("average score:", np.round(np.mean(l_topic_scores[l]), 3))
        else:
            pass
        
        if len(savepath)>0:
            savepath_lang = os.path.join(savepath, langs[l].name+"_words.txt")
            print("saving top words to :", savepath_lang)
            with open(savepath_lang, 'w', encoding='utf-8') as f:
                f.write("\n".join(top_topic_words))

    print("\n")

def calc_hgs(langs, l_word_topic_counts):
    v_scores = []

    if langs[0].name=="a":

        for l in range(len(langs)):
            word_topic_counts = np.asarray(l_word_topic_counts)[l]

            predict_topics = []
            truth_topics = []

            for i in range(word_topic_counts.shape[0]):
                
                # this is faulty. will always assume the first max even though there may be
                # more than 1 max
                try:
                    truth_topics.append(langs[l].ix2w[i][0]) # first digit is true topic
                    max_topic = np.argmax(word_topic_counts[i])
                    predict_topics.append(max_topic)
                except:
                    pass


#            hg_score = homogeneity_score(predict_topics, truth_topics)
            v_score = v_measure_score(predict_topics, truth_topics)
            print("vmeasure score:", v_score)
            v_scores.append(v_score)

    return v_scores

#top_words_f
#= "results/top_words/dev0/en-ro-gauss-ntopics100/stagger0-sharedParams1-temp1-interpolate0.5-scaling1/en_words.txt"
#



def investigate(topic_props_f, en_docs_f, top_words_f):
    topic_props = np.loadtxt(topic_props_f)
    test_data = pd.read_csv(en_docs_f, sep="\t", header=None, names=['query_id', 'topic', 'sentence'], encoding='utf-8')
    test_data = test_data.iloc[:,2].values

    with open(top_words_f, 'r') as f:
        top_words = f.readines()

    for i in range(3):
        sort_topic = np.argsort(topic_props[i])[::-1]
        print(test_data[i])
        for k in range(5):
            topic1 = sort_topic[k]
            print(topic_props[i][topic_1], top_words[topic1])
            print("--"*5)   

def external_score_top_words(wordsout_dir, score_type="npmi"):
    url = f"http://palmetto.aksw.org/palmetto-webapp/service/{score_type}?words="
    if os.path.exists(f"{wordsout_dir}/external_{score_type}.txt"):
#        print("file exists", wordsout_dir)
        return

    print("Running:", wordsout_dir)

    if not os.path.exists(os.path.join(wordsout_dir, 'en_words.txt')):
        print("en_words does not exist for", wordsout_dir)
        print("need to run top words from the model")
        return

    with open(os.path.join(wordsout_dir, 'en_words.txt'), 'r') as f:
        data = f.readlines()

    score = []
    for i, line in enumerate(data):
        print("reading line:", i)
        url = f"http://palmetto.aksw.org/palmetto-webapp/service/{score_type}?words="
        line = line.split()[2:12] # palmetto can only do up to 12
        rest_str = "%20".join(line)
        url = url + rest_str
        r = requests.get(url)
        time.sleep(10)
        score.append(float(r.content))

    score = np.array(score)
    np.savetxt(f"{wordsout_dir}/external_{score_type}.txt", score)
    print(wordsout_dir, np.around(np.mean(score),4))

def generate_tables(fd="", score_type=""):
    modes = {"0":"GAUSS", "1":"DISC", "0.5":"SMIX", "2":"ALDA"}

    lines = []
    for datasize in range(1000,  9000, 1000):
        line = {}
        for intp in ["0", "1", "0.5", "2"]: 
            split_score = [] 
            for split in range(5):
                if intp == "1":
                    fn = f"stagger0-sharedParams100-temp-interpolate0-scaling1/{datasize}.split{split}/external_{score_type}.txt"
                else:
                    fn = f"stagger0-sharedParams1-temp-interpolate{intp}-scaling1/{datasize}.split{split}/external_{score_type}.txt"
                if not os.path.exists(os.path.join(fd, fn)):
                    print("missing :", fn)  
                    continue 
                data = np.loadtxt(os.path.join(fd, fn))
                split_score.append(np.mean(data))

            mode = modes[intp]
            line[mode] = np.around(np.mean(split_score),4)
        line["datasize"] = datasize
        lines.append(line)

    results_df = pd.DataFrame(lines)
    results_df = results_df[["datasize", "DISC", "GAUSS", "SMIX", "ALDA"]]

    save_prefix = f"results/tables/{score_type}"
    print(f"write results to : {save_prefix} .md and .tex")
    with open(f'{save_prefix}.md', 'w') as f:
        f.write(results_df.to_markdown())
    with open(f'{save_prefix}.tex', 'w') as f:
        f.write(results_df.to_latex())

    #df.to_latex()
    #df.to_markdown()


if __name__ == "__main__":


    # change sys.argv[1] to scoring 
    #wordsout_dir = sys.argv[1]
    for score_type in ["cv", "npmi"]:
        #external_score_top_words(wordsout_dir, score_type=score_type)

        fd = "results/top_words/dev2/ne-split-gauss-ntopics20"
        generate_tables(fd=fd, score_type=score_type)




