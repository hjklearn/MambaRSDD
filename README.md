# MambaRSDD: Depth-Assisted Mamba with Contrastive Learning for Rail Surface Defect Detection
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SMH0o2tDe2CLvvIACA1-Ej2lRNDBRfyj?usp=sharing)
![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) 
![last commit](https://img.shields.io/github/last-commit/Sunnyhaze/IML-ViT)
![GitHub](https://img.shields.io/github/license/Sunnyhaze/IML-ViT?logo=license)
![](https://img.shields.io/github/repo-size/sunnyhaze/IML-ViT?color=green)
![](https://img.shields.io/github/stars/sunnyhaze/IML-ViT)
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/Sunnyhaze) 

This repo contains an official PyTorch implementation of our paper: [IML-ViT: Benchmarking Image Manipulation Localization by Vision Transformer.](http://arxiv.org/abs/2307.14863)

![overview](./images/overview.png)


## 📰1 News 
- [2025/01/02] 🎉🎉Please keep the updates coming！！！ 

## 2 Our environment
Ubuntu LTS 20.04.1 + CUDA 11.8 + Python 3.10 + PyTorch 2.0.0

Step1. Creating virtual environments
conda create -n mambarsdd python=3.10
conda activate mambarsdd
Step2. Installing torch, torchvision, cuda
conda install cudatoolkit==11.8 -c nvidia
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
Step3. Install packaging
conda install packaging
Step4. Entering the VMamba environment
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip3 install -r requirements.txt
cd kernels/selective_scan
pip3 install .
Step5. Modifying triton version
pip uninstall triton
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
Step6. Modifying numpy version
pip uninstall numpy
pip install numpy==1.26.3


## 3 Quick Start 
### 3.1 Training on your datasets
**Now training code for IML-ViT is released!**

### 3.2 Prepare NEU RSDDS-AUG Datasets
🎉🎉You can access the dataset in this paper:![https://img.shields.io/github/last-commit/Sunnyhaze/IML-ViT](https://ieeexplore.ieee.org/abstract/document/9769694)

### 3.3 Prepare the Pre-trained Weights
🎉🎉The Pre-training weights will be published！

### 3.4 Start Training Script
Still on Working...

### 3.5 Visualize the results
🎉🎉The results will be published！


## 4 Acknowledgement
The implement of this project is based on the codebases bellow. <br>
- [BBS-Net](https://github.com/zyjwuyan/BBS-Net) <br>
- [VMamba](https://github.com/MzeroMiko/VMamba) <br>
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) <br>

If you find this project helpful, Please also cite codebases above.
