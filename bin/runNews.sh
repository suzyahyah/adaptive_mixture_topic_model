#!/usr/bin/env bash

C=byobu-tmux
NT=20
DATA=ne-split
SP=100

#for DATA in en-ro en-fr en-fi en-de
# 0-Gaussian, 1-Discrete 2-smart interpolate

for INTP in 0; do #  0.5 2 #0 2 4 # 6 7
  for SPLIT in 0 1 2 3 4; do
    for TRAINSIZE in 1000 2000 3000 4000 5000 6000 7000 8000; do
      ts=${TRAINSIZE:0:1}

      DIR=${DATA}_${INTP}_${NT}/${TRAINSIZE}.${SPLIT}
      mkdir -p news_logs/qsub/${DIR}
      mkdir -p news_logs/qsub_e/${DIR}
      echo "Running: $RUNX, Done: $DONEX"

      echo ne1${INTP}${TRAINSIZE}.${SPLIT} missing

      qsub -N n${INTP}.${ts}.${SPLIT} -o news_logs/qsub/${DIR} -e \
      news_logs/qsub_e/${DIR} ./bin/runGaussian_py.sh $DATA $SP "${INTP}" 0 top_words $NT ${TRAINSIZE} $SPLIT

      #bash ./bin/runGaussian_py.sh $DATA $SP "${INTP}" 0 top_words $NT ${TRAINSIZE} $SPLIT

      sleep 10
    done
  done
done
