import base64
import json
from contextlib import asynccontextmanager
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from yolov9 import YOLOv9
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, items
    print("启动应用程序啦")
    model = YOLOv9(model_path="best.onnx", class_mapping_path="data/coco.yaml", original_size=(640, 640),
                   score_threshold=0.5, conf_thresold=0.5, iou_threshold=0.45, device="cpu")
    yield


app = FastAPI(lifespan=lifespan)


class Item(BaseModel):
    text: str

class Req(BaseModel):
    base64_str: str
@app.post("/get_base")
async def get_base(img_path: Item):
    print(img_path)
    # img = cv2.imread(img_path)
    # outputs = model.detect(img)
    with open(img_path.text, "rb") as f:
        byte_data = f.read()
    base64_str = base64.b64encode(byte_data).decode("utf-8")  # 二进制转base64
    data = Item(base64_str=base64_str)
    outputs = return_outputs(data)
    return {'outputs': outputs}

@app.post("/return_outputs")
async def return_outputs(req: Item):
    # 将base64编码的图片解码为原始图像
    image_base64 = req.text
    print("***********",image_base64)
    image_array = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    outputs = model.detect(image)
    # print(outputs)
    return {"succ": outputs}


if __name__ == '__main__':
    uvicorn.run(app="fastapi_v9:app", reload=True, host="*.*.*.*", port=8080)
