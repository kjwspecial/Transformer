# Transformer
PyTorch implementation of the Transformer.

German -> English translation task.


# ToDo
* Beam Search
* Weight Sharing

# Requirement
* Python
* Spacy
* Apex  
* torchtext
* Pytorch
* tqdm
* numpy

# Usage
### 1) Train the model - Datset : IWSLT 
```bash
python train.py -epoch 10 -batch 10 ...

python train.py
```
### 2) Test the model - Datset : Multi30k
```bash
python test.py
```
### 3) Greedy Decoding - Dataset : Multi30k
``` bash
python GreedyDecoder.py
```

# Reference
https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
