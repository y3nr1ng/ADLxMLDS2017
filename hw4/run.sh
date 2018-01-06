#!/usr/bin/env bash

# download and extract the model
curl -O https://www.csie.ntu.edu.tw/~b03902036/saved_model.zip
unzip -o saved_model.zip -d .
rm -f saved_model.zip

# execute
python3.5 generate.py $1
