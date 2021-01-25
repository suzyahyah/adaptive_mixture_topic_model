#usr/bin/env bash
export PYTHONIOENCODING=utf-8

#need to set the path so that we can find our pxd file
export PYTHONPATH=$PYTHONPATH:code/topicModelP/

declare -A M #model
declare -A H #hyperparams
declare -A L #logic
declare -A N #misc settings
declare -A D #directories
declare -A G #gen configs

GENDATA=0

N=(
["CYTHON"]=1
["TRAIN_TEST_VALID"]=$5
["SHORT_DOCS"]=1
)

if [ ${N["CYTHON"]} -eq 1 ]; then echo "compiling cython."; ./bin/setupcython.sh; fi
if [ $? -ne 0 ]; then echo "cython compile failed. exiting"; ./$0; fi 

M=(
["DATA"]=$1 #en-fi
["MODEL"]=gauss
["NTOPICS"]=$6
["NDIM"]=300
["NUM_LANG"]=2
["NITER"]=100 # iters to run
["LOAD_ITER"]=99 # where we load from 
["MAX_EVAL"]=201
)

H=(
["MU0_B"]=1
["COV0_SCALAR"]=0 # dev0 uses 1, dev2 uses empirical
["KMEANS"]=0
["SAMPLE_MEAN"]=1
["TRAIN_SIZE"]=$7
["SPLIT"]=$8
)

L=(
["SHARED_PARAMS"]=$2 # 100 is vanilla, 1 is shared params gaussian, 0 is not shared gaussian 
["INTERPOLATE"]=$3
["STAGGER"]=$4
["SCALING_DOF0"]=1 #1 is scale, 0 is dont scale. 
)

G=(
["NDIM"]=10
["NTOPICS"]=5
["NVOCAB_PER_TOPIC"]=100 #size of topic
["NDOCS_PER_TOPIC"]=50 #can calc vocab:docs-ratio
["COV0_SCALAR"]=1 # dev0 uses 1, dev2 uses empirical
)

SAVEDIR=${M["DATA"]}-${M["MODEL"]}-ntopics${M["NTOPICS"]}/stagger${L["STAGGER"]}-sharedParams${L["SHARED_PARAMS"]}-temp${H["TEMP"]}-interpolate${L["INTERPOLATE"]}-scaling${L["SCALING_DOF0"]} #-trainsize${H["TRAIN_SIZE"]}.split${H["SPLIT"]}

D=(
["PICKLE_DIR"]=$SAVEDIR
["TOPIC_PROP_DIR"]=results/topic_prop/$SAVEDIR
["TRAIN_PROP_DIR"]=results/train_topic_prob/$SAVEDIR
["LLH_DIR"]=results/logllh/$SAVEDIR
["HGS_DIR"]=results/hgs/$GENCONFIG
["WORDSOUT_DIR"]=results/top_words/$SAVEDIR
["ENTROPY_DIR"]=results/entropy/$SAVEDIR
["TRECIN_DIR"]=results/ranking/$SAVEDIR
["TRECOUT_DIR"]=results/trec_score/$SAVEDIR
)


PY="python code/topicModelP/main.py"

printf "\n== Misc ==\n"
for var in "${!N[@]}"
do
  PY+=" --${var,,} ${N[$var]}"
  echo "| $var:${N[$var]}"
done

printf "\n== Hyperparameters ==\n"
for var in "${!H[@]}"
do
  PY+=" --${var,,} ${H[$var]}"
  echo "| $var:${H[$var]}"
done

printf "\n== Model ==\n"
for var in "${!M[@]}"
do
  PY+=" --${var,,} ${M[$var]}"
  echo "| $var:${M[$var]}"
done

printf "\n== Logic ==\n"
for var in "${!L[@]}"
do
  PY+=" --${var,,} ${L[$var]}"
  echo "| $var:${L[$var]}"
done

printf "\n== Directory ==\n"
for var in "${!D[@]}"
do
  PY+=" --${var,,} ${D[$var]}"
  echo "| $var:${D[$var]}"
  mkdir -p ${D[$var]}
done

echo $SAVEDIR

echo $PY
eval $PY

subject="${@}"
