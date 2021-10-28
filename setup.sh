#!/bin/bash
echo Downloading Dataset

wget -nc http://www.openslr.org/resources/12/train-clean-100.tar.gz

tar -xf train-clean-100.tar.gz

mkdir models

echo Downloading Virtualenv

brew install virtualenv

virtualenv venv

source venv/bin/activate

echo Install all dependencies with Pip

pip install -r requirements.txt