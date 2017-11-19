# HW2 Video Captioning
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
./hw2_seq2seq.sh [data dir] [output filename] [peer review filename]
```
