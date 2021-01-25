#### Basic Run

To compile cython files:
> `python code/setup.py build_ext --inplace`

To preprocess embeddings for faster loading, do
> `python code/topicModelP/preprocess_en_embed.py`


Basic 20 newsgroup vanilla (discrete LDA)
> `bash ./bin/runGaussian_py.sh $DATA $SHARED_PARAMS $INTERPOLATE $STAGGER $TRAIN_MODE $NTOPICS $TRAIN_SIZE $SPLIT `

Batch run for sungrid engine
> `bash ./bin/runNews.sh`
> `bash ./bin/runWiki.sh`

#### Other available configs
See `./bin/runGaussian_py.sh` which contains full list of configs and descriptions

For the multi-lingual case, a config file with data_paths should be provided in `configs/data_paths.yaml`. See example in `configs/data_path.yaml`


#### External Evaluations
External evaluations (cv, npmi) relies on the [Palmetto library](https://github.com/dice-group/Palmetto). We only conduct external scoring against the monolingual dataset. This code relies on `en_words.txt` being present in the `DIR`. 
  
> bash ./bin/external_score.sh 

