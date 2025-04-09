## Mamba-YOLO: Multi-Level Adaptive Rectangular Convolution for Document Layout Analysis
👉 [Click here to view the English instructions](./README.md)
## 介绍

​	本文介绍了一种基于中文多种类文档的版面分析数据集和一种新颖的版面分析方法。在数据集方面，我们将现有工作中的‘Title’类别进行细化，同时引入了多个细粒度下的标题类别信息；其次我们将图片，表格和公式的标注范围进行调整，使其能够覆盖对应的描述信息或者公式序号。在模型方法方面，我们介绍了一种基于YOLOv9框架，引入Mamba结构和多层自适应矩形卷积方法，有效的提升了文档版面分析的工作。

​	我们的方法在现有工作中，相较于YOLOv9有了一定的性能提升，但是距离SOTA仍然有改进空间。

​	我们将会持续发布最新的资讯和更新的权重信息，以及我们最新的研究成果。

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

### 5.其他环境

其他环境参考YOLOv9中环境

## 数据集介绍

**PeKi**主要包含十四种文档版面分析的类别，主要有：

Formula, Formula-Num, Figure, Figure-Caption, Reference, Table, Table-Caption, Footer, Header, Doc-Title, Title-Id, Title-NoId, Title-Body, Title-Last.

我们公开了我们论文中的相关数据集[PeKi](https://huggingface.co/datasets/Mwk19990801/PeKi)示例，可以供用户更好的去了解我们详细的内容。

如果想要更详细的信息，请将申请报告发送至邮箱，我们收到后将会与您联系。

## 训练过程

Multiple GPU training

```
# train DocMY models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train_dual.py --workers 4 --device 0,1,2,3 --sync-bn --batch -1 --data data/peki.yaml --img 640 --cfg models/detect/DocMY.yaml --weights '' --name DocMY --hyp hyp.scratch-high.yaml --min-items 0 --epochs 100 --close-mosaic 15
```

## 验证过程

```
# evaluate converted yolov9 models
python val.py --data data/peki.yaml --img 640 --batch 8 --conf 0.001 --iou 0.7 --device 0 --weights './DocMY.pt' --save-json --name yolov9_peki_val
```

## 推理过程

```
# inference converted yolov9 models
python detect.py --source './data/document/notice.jpg' --img 640 --device 0 --weights './DocMY.pt' --name yolov9_peki_detect
```

我们同时还公布了我们使用YOLOv9训练后的模型权重和代码，可以参考yolov9-onnxruntime中的内容，并完成了服务端和请求端的代码信息。

## TODO

1.验证不同模型在其他开源数据集的实验，更好的补充我们文章。

- [ ] PubLayNet
- [ ] CDLA
- [ ] D4LA
- [ ] DocLayNet

2.我们正在收集并进行标注更多的文档版面分析数据集，将会及时的在本地公布并更新数据集网址。

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics), [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) and [VMamba](https://github.com/MzeroMiko/VMamba). The selective-scan from [Mamba](https://github.com/state-spaces/mamba).
