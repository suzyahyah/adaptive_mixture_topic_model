#!/usr/bin/env bash

NT=20
DATA=ne-split
SP=100

#for DATA in en-ro en-fr en-fi en-de
# 0-Gaussian, 1-Discrete 2-smart interpolate

for INTP in 0;  do #  0.5 2 #0 2 4 # 6 7
  for SPLIT in 0 1 2 3 4; do
    for TRAINSIZE in 1000 2000 3000 4000 5000 6000 7000 8000; do
      ts=${TRAINSIZE:0:1}

      #DIR=${DATA}_${INTP}_${NT}/${TRAINSIZE}.${SPLIT}
      DIR=results/top_words/dev2/ne-split-gauss-ntopics20/stagger0-sharedParams100-temp-interpolate${INTP}-scaling1/$TRAINSIZE.split$SPLIT

      python code/topicModelP/report_utils.py $DIR
#      ls $DIR/external_npmi.txt
      
    done
  done
done
