# HW1 Sequence Labeling
## Environment
The scripts were tested under Windows 7 64-bit, using CUDA 8 R2 with cuDNN 6.1 with Tesla P100 PCI-E 16GB.

## Quick start
1. Create a `conda` environment through
```
conda create --name adl --file package_list.txt
```
assuming user already have a functional CUDA environment.

2. Activate the environment 
```
activate adl
```
for power users in Windows our *nix environment, please use
```
source activate adl
```
If your environment is not configured Keras to use TensorFlow for any reason, please do so by setting the environment variable
```
KERAS_BACKEND=tensorflow
```
for your shell after the activation of the `conda` environment.

3. Run the script!
```
./hw1_best.sh [input directory] [output filename]
```
or any of the following: `hw1_cnn.sh`, `hw1_rnn.sh`
