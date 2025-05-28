import base64

import cv2
import requests
import json
img_path = r"assets/1.png"
with open(img_path, 'rb') as file:
    image_data = file.read()
# base64 encode
url2 = "http://*.*.*.*:8090/return_outputs"
data = {
    "image_base64": str(base64.b64encode(image_data), encoding="utf-8")
}
response = requests.post(url2, data=json.dumps(data))
content = response.content
result = json.loads(content)
print(result)