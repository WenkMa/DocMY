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

- [x] PubLayNet

- [x] CDLA

- [x] D4LA

- [x] DocLayNet

  | Datasets  | Method      | P(%)       | R(%)       | mAP50(%)   | mAP50-95(%) |
  | --------- | ----------- | ---------- | ---------- | ---------- | ----------- |
  | PubLayNet | VGT(SOTA)   | -          | -          | 98.1       | 96.2        |
  |           | Layoutlmv3  | -          | -          | 98.1       | 95.1        |
  |           | DiT         | -          | -          | 97.9       | 93.8        |
  |           | YOLOv9      | 87.2       | 86.5       | 88.1       | 83.1        |
  |           | VMamba      | 97.4       | 95.3       | 88.9       | 90.78       |
  |           | DocMY(ours) | 92.4(+5.2) | 91.5(+5.0) | 93.3(+5.2) | 88.4(+5.3)  |
  | CDLA      | Layoutlmv3  | -          | -          | 66.9       | 47.0        |
  |           | YOLOv5      | 91.5       | 85.7       | 91.9       | 66.6        |
  |           | YOLOv8      | 90.2       | 88.2       | 93.8       | 77.2        |
  |           | YOLOv9      | 90.1       | 87.4       | 94.0       | 77.3        |
  |           | VMamba      | 89.5       | 88.0       | 93.5       | 78.3        |
  |           | DocMY(ours) | 93.2(+2.2) | 91.4(+4.0) | 96.1(+2.1) | 83.3(+6.0)  |
  | DocLayNet | GLAM(SOTA)  | -          | -          | -          | 80.8        |
  |           | Layoutlmv3  | -          | -          | 90.2       | 72.6        |
  |           | YOLOv9      | 88.5       | 81.8       | 89.6       | 69.8        |
  |           | VMamba      | 88.6       | 84.0       | 91.1       | 69.8        |
  |           | DocMY(ours) | 89.5(+1.0) | 81.8       | 90.2(+0.6) | 70.9(+1.1)  |
  | D4LA      | VGT(SOTA)   | -          | -          | 81.9       | 68.8        |
  |           | Layoutlmv3  | -          | -          | 75.2       | 61.9        |
  |           | YOLOv9      | 75.1       | 64.1       | 69.8       | 56.0        |
  |           | VMamba      | 77.4       | 66.4       | 71.7       | 57.8        |
  |           | DocMY(ours) | 77.8(+2.7) | 71.7(+7.6) | 76.7(+6.9) | 62.8(+6.8)  |

  Effects of Our Method on PubLayNet, CDLA, DocLayNet, and D4LA. Bold indicates performance improvement compared to baseline YOLOv9. - indicates that we did not find or reproduce the result.

2. We have collected documents in more languages and are currently marking them. Meanwhile, we have also uploaded our latest weights to Hugging Face. We will update the progress of our work in time.

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics), [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO), [VMamba](https://github.com/MzeroMiko/VMamba) and [Unlim](https://github.com/microsoft/unilm). The selective-scan from [Mamba](https://github.com/state-spaces/mamba).
