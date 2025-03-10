# YOLOv9 ONNXRuntime (Python)

Performing Object Detection for YOLOv9 with ONNXRuntime

![ONNXRUNTIME YOLOv9 Object Detection](https://github.com/Reversev/yolov9-onnxruntime/blob/main/assets/horses.jpg)


## Installation

```shell
git clone https://github.com/danielsyahputra/yolov9-onnxruntime.git
cd yolov9-onnxruntime
pip install -r requirements.txt
```

```requirements.txt``` partly comes from [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9).


### ONNX Runtime
Use Nvidia GPU:
```pip install onnxruntime-gpu```

Otherwise:
```pip install onnxruntime```

### DOWNLOAD Yolov9 ONNX file
```python
python export.py --weights ./ckpt/yolov9-c.pt --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --include onnx
```
Link: [yolov9-c.onnx(pwd:smus)](https://pan.baidu.com/s/1qQuKyhFsmduaWuBr_02ZQw?pwd=smus) and [yolov9-e.onnx(pwd=pwza)](https://pan.baidu.com/s/1_QRAQEIq4ki8Xdu5M2E7IQ? 
).

Download onnx model and save weights directory.

### Inference on Image with CPU/GPU

(1) CPU: need onnxruntime

```python
pip install onnxruntime
python main.py --source inference/images/horse.jpg --weights weights/yolov9-c.onnx --classes weights/coco_names.yaml --image
```

(2) GPU: need onnxruntime-gpu
```python
pip install onnxruntime-gpu
python main.py --source inference/images/horse.jpg --weights weights/yolov9-c.onnx --classes weights/coco_names.yaml --image --device cuda
```


### Inference on Video

(1) CPU: need onnxruntime
```
pip install onnxruntime
python main.py --source inference/video/demo.mp4 --weights weights/yolov9-c.onnx --classes data/coco_names.yaml --video
```

(2) GPU: need onnxruntime-gpu
```
pip install onnxruntime-gpu
python main.py --source inference/video/demo.mp4 --weights weights/yolov9-c.onnx --classes data/coco_names.yaml --video --device cuda
```

# References:
[1] [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)

[2] [https://github.com/danielsyahputra/yolov9-onnx/tree/master](https://github.com/danielsyahputra/yolov9-onnx/tree/master)

## License / 许可
[MIT License](LICENSE)
