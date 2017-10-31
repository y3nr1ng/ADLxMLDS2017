#!/usr/bin/env bash

COMMAND=""
if [ "$1" == "-h" ]; then
    echo "Usage: `basename $0` [data dir] [output filename] [peer review filename]"
    exit 0
fi

# use default instruction
if [ "$#" -eq "0" ]; then
    python model_seq2seq.py "data" "train" -m train -r
    exit;
fi

# test set
python model_seq2seq.py $1 "test" -o $2
# peer review
python model_seq2seq.py $1 "review" -o $2
