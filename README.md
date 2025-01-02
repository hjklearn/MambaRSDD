# MambaRSDD: Depth-Assisted Mamba with Contrastive Learning for Rail Surface Defect Detection
![Powered by](https://img.shields.io/badge/Based_on-Pytorch-blue?logo=pytorch) 
![last commit](https://img.shields.io/github/last-commit/hjklearn/MambaRSDD)
![GitHub](https://img.shields.io/github/license/hjklearn/MambaRSDD?logo=license)
![](https://img.shields.io/github/repo-size/hjklearn/MambaRSDD?color=green)
![](https://img.shields.io/github/stars/hjklearn/MambaRSDD)
[![Ask Me Anything!](https://img.shields.io/badge/Official%20-Yes-1abc9c.svg)](https://GitHub.com/hjklearn) 
<br>
- ## This repo contains an official PyTorch implementation of our paper: <br>
- MambaRSDD: Depth-Assisted Mamba with Contrastive Learning for Rail Surface Defect Detection.



## 📰1 News 
- [2025/01/02] 🎉🎉Please keep the updates coming！！！ 

## 2 Our environment
Ubuntu LTS 20.04.1 + CUDA 11.8 + Python 3.10 + PyTorch 2.0.0

### The installation steps are as follows：
```
conda create -n mambarsdd python=3.10
conda activate mambarsdd
conda install cudatoolkit==11.8 -c nvidia
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install packaging
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip3 install -r requirements.txt
cd kernels/selective_scan
pip3 install .
pip uninstall triton
pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
pip uninstall numpy
pip install numpy==1.26.3
```

## 3 Quick Start 
### 3.1 Prepare NEU RSDDS-AUG Datasets
🎉🎉You can access the dataset in this paper:![(https://ieeexplore.ieee.org/abstract/document/9769694)]

### 3.2 Prepare the Pre-trained Weights
🎉🎉The pre-trained weights from VMamba-Tiny are initialized before the model training begins. You can obtain it from:[pth_vmamba_tiny](https://pan.baidu.com/s/1ky8Ye_NYLV4FtIMPziu2xw)

### 3.3 Model Training and Inference
#### RGBD Rail Surface Defect Detection
To train MambaRSDD for Rail Surface Defect Detection on NEU RSDDS-AUG Datasets, use the following commands for different configurations:
```
cd MambaRSDD/RSDD_Tool
python train.py
```
If you only want to test the performance:
```
cd MambaRSDD/RSDD_Tool
python test.py
```
#### RGBD Salient Object Detection
To train MambaRSDD for Salient Object Detection on part of NJU2K+NLPR Datasets, use the following commands for different configurations:
```
cd MambaRSDD/SOD_Tool
python train.py
```
If you only want to test the performance:
```
cd MambaRSDD/SOD_Tool
python test.py
```
### 3.4 Visualize the results
🎉🎉The visualization results be published：
![A comparison of the visualization results from several methods, with the method proposed in this paper highlighted in bold.]([https://github.com/yourusername/yourrepo/blob/main/images/your-image.png](https://github.com/hjklearn/MambaRSDD/blob/main/Fig.%208.png)?raw=true)


## 4 Acknowledgement
The implement of this project is based on the codebases bellow. <br>
- [BBS-Net](https://github.com/zyjwuyan/BBS-Net) <br>
- [VMamba](https://github.com/MzeroMiko/VMamba) <br>
- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) <br>

If you find this project helpful, Please also cite codebases above.
