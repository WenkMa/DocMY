## Mamba-YOLO: Multi-Level Adaptive Rectangular Convolution for Document Layout Analysis

👉 [Click here to view the 中文文档说明](./readme_zh.md)

## Introduction

​	This paper presents a document layout analysis dataset for various types of Chinese documents and a novel layout analysis method. On the dataset side, we refine the existing "Title" category and introduce multiple subcategories under a finer granularity. We also adjust the annotation ranges for images, tables, and formulas to include descriptive texts and formula indices. On the model side, we propose an approach based on the YOLOv9 framework by integrating the Mamba architecture and Multi-Level Adaptive Rectangular Convolutions, significantly improving document layout analysis performance.

​	Compared to YOLOv9, our method achieves a noticeable performance improvement, although there is still room for improvement before reaching the SOTA. 

​	We will continue releasing the latest news, updated weights, and our newest research findings.

## Environment Setup

### 1.  Installation

DocMY is developed based on `torch==2.1.0` `transformers==4.49.0` and `CUDA Version==11.8`

### 2. Clone Project

```
git clone https://github.com/WenkMa/DocMY.git
```

### 3. Create and activate a conda environment.

```
conda create -n DocMY -y python=3.10
conda activate DocMY
```

### 4. Install torch

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 5. Other dependencies

Refer to the YOLOv9 environment setup for additional dependencies.

## Dataset Description

**PeKi** includes 14 categories for document layout analysis, namely:

Formula, Formula-Num, Figure, Figure-Caption, Reference, Table, Table-Caption, Footer, Header, Doc-Title, Title-Id, Title-NoId, Title-Body, Title-Last.

We have released a sample of the dataset used in our paper, [PeKi](https://huggingface.co/datasets/Mwk19990801/PeKi), to help users better understand our work.

For more detailed information, please send an application request via email. We will contact you upon receiving it.

## Training Procedure

Multiple GPU training

```
# train DocMY models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train_dual.py --workers 4 --device 0,1,2,3 --sync-bn --batch -1 --data data/peki.yaml --img 640 --cfg models/detect/DocMY.yaml --weights '' --name DocMY --hyp hyp.scratch-high.yaml --min-items 0 --epochs 100 --close-mosaic 15
```

## Evaluation Procedure

```
# evaluate converted yolov9 models
python val.py --data data/peki.yaml --img 640 --batch 8 --conf 0.001 --iou 0.7 --device 0 --weights './DocMY.pt' --save-json --name yolov9_peki_val
```

## Inference Procedure

```
# inference converted yolov9 models
python detect.py --source './data/document/notice.jpg' --img 640 --device 0 --weights './DocMY.pt' --name yolov9_peki_detect
```

We also provide trained YOLOv9 model weights and code. For more information, refer to the `yolov9-onnxruntime` implementation, which includes both server-side and client-side code.

## TODO

1. Validate different models on other open-source datasets to better support our paper:

- [ ] PubLayNet
- [ ] CDLA
- [ ] D4LA
- [ ] DocLayNet

2. We are currently collecting and annotating more document layout analysis datasets, which will be released and updated in a timely manner.。

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics), [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) and [VMamba](https://github.com/MzeroMiko/VMamba). The selective-scan from [Mamba](https://github.com/state-spaces/mamba).
