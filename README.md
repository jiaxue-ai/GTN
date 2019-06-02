This is the official PyTorch implementation of the paper:

```
[Gated Transfer Network for Transfer Learning](https://arxiv.org/abs/1810.12521)
[Yi Zhu](https://sites.google.com/view/yizhu/home) and [Jia Xue](http://jiaxueweb.com/) and [Shawn Newsam](http://faculty.ucmerced.edu/snewsam/)
ACCV 2018
```

# Installation

We recommend using a Conda environment. We use PyTorch 1.1, CUDA 9.0 and python 3.7.  

```bash
conda create -n gtn python=3.7
conda activate gtn
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install easydict
``` 

# Data Preparation

Please see [datasets README](https://github.com/jiaxue1993/GTN/tree/master/dataset) for more details. 

# Experiments 

We take XXX as an example. 


- train

- evaluate


Other experiments are similar except some hyper-parameter changes. 


# Citation

If you use this code for your research, please consider citing our paper:

```
@inproceedings{zhu2018GTN,
  author    = {Yi Zhu and Jia Xue and Shawn Newsam},
  title     = {Gated Transfer Network for Transfer Learning},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2018}
}
```