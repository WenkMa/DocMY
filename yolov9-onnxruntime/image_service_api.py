import json

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import cv2
import base64
import uvicorn

from yolov9 import YOLOv9




class Req(BaseModel):
    image_base64: str

app = FastAPI()


@app.post("/upload_image")
async def upload_image(req: Req):
    image_base64 = req.image_base64
    image_array = np.frombuffer(base64.b64decode(image_base64), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # cv2.imwrite("tmp.jpg", image)
    h,w,_=image.shape
    model = YOLOv9(model_path="best.onnx", class_mapping_path="data/coco.yaml", original_size=(w, h),
                   score_threshold=0.5, conf_thresold=0.5, iou_threshold=0.45, device="cpu")
    detections = model.detect(image)
    return {
            "result":detections
        }


if __name__ == "__main__":
    uvicorn.run("image_service_api:app", host="0.0.0.0", port=10000, reload=True)
