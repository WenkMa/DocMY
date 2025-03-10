import base64
import cv2
import requests
import json


def draw_detections(img, detections, img_path):
    """
    Draws bounding boxes and labels on the input image based on the detected objects.

    Args:
        img: The input image to draw detections on.
        detections: List of detection result which consists box, score, and class_ids
        box: Detected bounding box.
        score: Corresponding detection score.
        class_id: Class ID for the detected object.

    Returns:
        None
    """

    for detection in detections:
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = [int(x) for x in detection['box']]
        class_id = detection['class_index']
        confidence = detection['confidence']
        class_name = detection["class_name"]
        # Retrieve the color for the class ID
        color = (255, 0, 0)

        # Draw the bounding box on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Create the label text with class name and score

        # label = f"{self.classes[class_id]}: {confidence:.2f}"
        label = f"{class_name}: {confidence:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(img_path, img)


with open("assets/1.png", "rb") as f:
    image = f.read()

data = {
    "image_base64": str(base64.b64encode(image), encoding="utf-8")
}
url = "http://*.*.*.*:10071/upload_image"
response = requests.post(url, data=json.dumps(data))
result = json.loads(response.content)
detections = result["result"]
print(detections)
# draw_detections(cv_img, detections, "tmp.png")
