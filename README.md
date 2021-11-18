# Contrastive Predictive Coding
## This repo contains the code by group 18 in the course DD2430 Project Course in Data Science
### KTH Royal Institute of Technology
### Fall 2021
<br />

The code is based on the following repository: https://github.com/gbarello/CPCLibriSpeech. 

<br />

### The code is divided into three branches:
* dataloader, containing files for the dataloading and initial attempts of adapting the code to work on the MEG dataset.
* gpu, containing the modifications in the code to allow for training on our GPU.
* stat_test, containing the statistical tests that were performed.

To setup, run the following:
```
$./setup.sh
$python train.py
$python test.py ./models/{model_timestamp}
```
