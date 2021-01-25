#!/usr/bin/python
# Author: Suzanna Sia
import os
from sklearn.datasets import fetch_20newsgroups

data_folder = "/home/ssia/projects/crossLing_topic_IR/data"

def load():
    
    select_topics = ["alt.atheism", "comp.graphics", "rec.motorcycles", "sci.med",
    "talk.politics.mideast"];

    news_train = fetch_20newsgroups(data_home=data_folder, subset="train",
            categories=categories, remove=('headers', 'footers', 'quotes'))





if __name__ == "__main__":
    load()
