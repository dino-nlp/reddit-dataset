<div align="center">

# Reddit dataset for Multi-task NLP

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<a href="https://www.kaggle.com/datasets/amangoyl/reddit-dataset-for-multi-task-nlp"><img alt="Dataset" src="https://img.shields.io/badge/Kaggle-dataset-red"></a><br>
</div>

## Description

- A multi-label dataset of Reddit posts having Suicidal and Sentiment labels.
- Techniques used: 
  - Transfer Learning with BERT, 
  - Multi-task Learning,
  - Pytorch Lightning,
  - Hydra,
  - Aim/Mlflow
  

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/dino-nlp/reddit-dataset.git
cd reddit-dataset

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/dino-nlp/reddit-dataset.git
cd reddit-dataset

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Results
TODO