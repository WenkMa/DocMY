import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import uvicorn

from yolov9 import YOLOv9


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, items
    print("启动应用程序啦")
    model = YOLOv9(model_path="best.onnx", class_mapping_path="data/coco.yaml", original_size=(640, 640),
                   score_threshold=0.5, conf_thresold=0.4, iou_threshold=0.45, device="cpu")
    yield


class Req(BaseModel):
    image_base64: str


app = FastAPI(lifespan=lifespan)


@app.post("/upload_image")
async def upload_image(req: Req):
    image_base64 = req.image_base64
    image_array = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # cv2.imwrite("tmp.jpg", image)
    h, w, _ = image.shape
    # model = YOLOv9(model_path="best.onnx", class_mapping_path="data/coco.yaml", original_size=(w, h),
    #                score_threshold=0.5, conf_thresold=0.5, iou_threshold=0.45, device="cpu")
    model.image_width, model.image_height = w, h
    detections = model.detect(image)
    # print(detections)
    # result=[]
    for detection in detections:
        #{'class_index': 3, 'confidence': 0.9331519, 'box': [203.167724609375, 575.1439819335938, 2210.716552734375, 2605.137451171875], 'class_name': 'table'}
        detection["class_index"]=int(detection["class_index"])
        detection["confidence"]=float(detection["confidence"])
        detection['box']= detection['box'].tolist()
    # print(detections)
    # print(type(detections))
    return {
            "result":detections
        }

if __name__ == "__main__":
    uvicorn.run("image_service_api_2:app", host="0.0.0.0", port=10071, reload=True)
