#!/usr/bin/python
# Author: Suzanna Sia

import string

stopwords_path = "assets/stoplists/en.txt"

class Preprocessor():
    def __init__(self, name):
        self.name = name
        print("processor name:", name)

        if self.name=="a" or self.name=="b":
            self.name="en"

        self.strip_punct = str.maketrans("", "", string.punctuation)
        self.strip_digit = str.maketrans("", "", string.digits)
     
        sw_path = "assets/stoplists/{}.txt".format(self.name)
        self.sw = self.get_sw(sw_path=sw_path)
        #self.sw = self.get_sw()

    def get_sw(self, sw_path=""):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords =f.readlines()
        stopwords = [w.strip() for w in stopwords]
        print("Reading en stopwords from :", stopwords_path)

        if not sw_path is None:
            with open(sw_path, 'r', encoding='utf-8') as f:
                stopwords2 = f.readlines()
            print("Reading stopwords from :", sw_path)
            stopwords2 = [w.strip() for w in stopwords2]

        stopwords.extend(stopwords2)
        stopwords = set(stopwords)
        return stopwords

    def rem_stopwords(self, words):
        stripped_words = []

        for w in words:
            w = w.strip()
            
            if w not in self.sw and not w.isdigit() and len(w)>2:
                stripped_words.append(w)

        return stripped_words


    def clean(self, words):
        keep_words = []
        for word in words.split():
            if word.count('@')>0 and word.count('.')>=0: # . is in the middle of word
                pass
            else:
                keep_words.append(word)

        words = " ".join(keep_words)
        words = words.translate(self.strip_punct).translate(self.strip_digit)
        #words = " ".join([word.translate(self.strip_punct).translate(self.strip_digit) for word in words.split()])
        #words = " ".join([word.translate(self.table) for word in words.split()])
        return words
        
