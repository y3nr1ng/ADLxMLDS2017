#!/usr/bin/env bash

if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` [data dir] [output filename] [peer review filename]"
  exit 0
fi

python model_seq2seq.py
