# FedTracker

This is the official code of paper "FedTracker: Furnishing Ownership Verification and Traceability for Federated Learning Model".

## Getting Start

First, create a virtual environment using Anaconda.
``` 
conda create -n fedtracker python=3.8
conda activate fedtracker
```

Second, you need to install the necessary packages to run FedTracker.
```
conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install geneal
pip install quadprog
pip install tqdm
```

After that, running the bash scripts in /bash
```
bash ./script/vgg16.sh
```

## Citing this work

If you use this repository for academic research, we highly encouraged you to cite our paper.
```
@article{shao2024fedtracker,
  title={Fedtracker: Furnishing ownership verification and traceability for federated learning model},
  author={Shao, Shuo and Yang, Wenyuan and Gu, Hanlin and Qin, Zhan and Fan, Lixin and Yang, Qiang and Ren, Kui},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2024}
}
```
