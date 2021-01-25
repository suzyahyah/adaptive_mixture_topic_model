#!/usr/bin/env bash

NT=20
DATA=en-fr
SPLIT=1
SP=100 

for DATA in en-ro en-fr en-pl en-es 
do
  for INTP in 1 # 0 0.5 2
  do
    for SPLIT in 0 1 2 3 4
    do
      for TRAINSIZE in 1000 7000
      do
        ts=${TRAINSIZE:0:1}
        LANG=${DATA:3:5}

        DIR=${DATA}_${INTP}_${NT}/${TRAINSIZE}.${SPLIT}

        mkdir -p wiki_logs/qsub/${DIR}
        mkdir -p wiki_logs/qsub_e/${DIR}

        echo ${LANG}${INTP}.${ts}.${SPLIT} missing
        qsub -N ${LANG}${INTP}.${ts}.${SPLIT} -o wiki_logs/qsub/${DIR} -e \
          wiki_logs/qsub_e/${DIR} ./bin/runGaussian_py.sh $DATA $SP "${INTP}" 0 train $NT \
        ${TRAINSIZE} $SPLIT
        sleep 10
      done
    done
  done
done 
