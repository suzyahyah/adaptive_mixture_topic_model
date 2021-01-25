#!/usr/bin/env bash
##
# This script gets muse multi-lingual embeddings

#wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.ro.vec
#wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec
#wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fi.vec
#wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.de.vec
#wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec

wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec
wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.pl.vec
wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.tr.vec
wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.sv.vec

mv *.vec assets/embeddings/
