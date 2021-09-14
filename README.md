# <img src=".github/README/icon.png" width="22"> SiamBOMB

[![License](https://img.shields.io/github/license/JackieZhai/SiamBOMB)](https://github.com/JackieZhai/SiamBOMB/blob/20210919/LICENSE)
![RepoSize](https://img.shields.io/github/repo-size/JackieZhai/SiamBOMB)
![CodeSize](https://img.shields.io/github/languages/code-size/JackieZhai/SiamBOMB)
[![Release](https://img.shields.io/github/v/release/JackieZhai/SiamBOMB?include_prereleases&sort=semver)](https://github.com/JackieZhai/SiamBOMB/releases)
[![Commit](https://img.shields.io/github/last-commit/JackieZhai/SiamBOMB)](https://github.com/JackieZhai/SiamBOMB/commits/20210919)

This repo is the second preview version of SiamBOMB, which is updating in September 2021.\
Copyright \(c\) 2021 Institute of Automation, Chinese Academy of Sciences. 
All rights reserved.
<p align="center"><img src=".github/README/affiliation.png" width="500"></p>

## Introduction
Our Paper (IJCAI 2020 Demo Track): [10.24963/ijcai.2020/776](https://www.ijcai.org/Proceedings/2020/0776.pdf)

<p align="center"><img src=".github/README/demo.png" width="500"></p>

1. TODO

## Setup
### 1. Configure environments
* Linux (Ubuntu 18.04) or Windows (10).
* GPU (at least have 4 GB memory).
* CUDA 10.1, 10.2, 11.1, etc. (with cuDNN).
* Anaconda 4.8+ (or virtualenv etc.) and Python 3.6+.
* C++ build tools (g++ in Linux or 2015+ in Windows).
* Download .zip or git clone our code.
2. Install dependencies
### 2. Install dependencies
```Shell
# TODO
```
### 3. Equip models
A simple SiamMask_E pretrained model: \
[Google Drive](https://drive.google.com/file/d/1VVpCAUJeysyRWdLdfW1IsT3AsQUQvwAU/view), [Baidu Pan](https://pan.baidu.com/s/1q64A2jPEWmdj264XrfvhBA) (key: jffj) \
You need to copy it into `pysot/experiments/siammaske_r50_l3/`.

## Citation
```
@inproceedings{SiamBOMB,
  title     = {SiamBOMB: A Real-time AI-based System for Home-cage Animal Tracking, Segmentation and Behavioral Analysis},
  author    = {Chen, Xi and Zhai, Hao and Liu, Danqian and Li, Weifu and Ding, Chaoyue and Xie, Qiwei and Han, Hua},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  pages     = {5300--5302},
  year      = {2020},
  month     = {7},
  doi       = {10.24963/ijcai.2020/776},
  url       = {https://doi.org/10.24963/ijcai.2020/776},
}
```

## References
```
@article{A_Common_Hub,
  title={A common hub for sleep and motor control in the substantia nigra},
  author={Liu, Danqian and Li, Weifu and Ma, Chenyan and Zheng, Weitong and Yao, Yuanyuan and Tso, Chak Foon and Zhong, Peng and Chen, Xi and Song, Jun Ho and Choi, Woochul and others},
  journal={Science},
  volume={367},
  number={6476},
  pages={440--445},
  year={2020},
  publisher={American Association for the Advancement of Science}
}
```
