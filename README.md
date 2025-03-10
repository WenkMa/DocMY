## Mamba-YOLO: Multi-Level Adaptive Rectangular Convolution for Document Layout Analysis



## 介绍

​	本文介绍了一种基于中文多种类文档的版面分析数据集和一种新颖的版面分析方法。在数据集方面，我们将现有工作中的‘Title’类别进行细化，同时引入了多个细粒度下的标题类别信息；其次我们将图片，表格和公式的标注范围进行调整，使其能够覆盖对应的描述信息或者公式序号。在模型方法方面，我们介绍了一种基于YOLOv9框架，引入Mamba结构和多层自适应矩形卷积方法，有效的提升了文档版面分析的工作。

​	我们的方法在现有工作中，相较于YOLOv9有了一定的性能提升，但是距离SOTA仍然有改进空间。

## 环境配置

### 1. Installation

DocMY is developed based on `torch==2.1.0` `transformers==4.49.0` and `CUDA Version==11.8`

### 2.Clone Project

```
git clone https://github.com/WenkMa/DocMY.git
```

### 3.Create and activate a conda environment.

```
conda create -n DocMY -y python=3.10
conda activate DocMY
```

### 4.Install torch

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```



## 数据集介绍

## 训练过程

## 验证过程

## 推理过程

## TODO



## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics) and [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO). The selective-scan from [Mamba](https://github.com/state-spaces/mamba).
