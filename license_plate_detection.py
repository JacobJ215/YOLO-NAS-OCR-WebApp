import cv2
from super_gradients.training import models
import numpy as np
import math
import easyocr

reader = easyocr.Reader(['en'], gpu=True)
def ocr_image(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)

    print(result)
    text = ""
    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1])>6 and res[2] >0.2:
            text = res[1]
    return str(text)


cap = cv2.VideoCapture("./Videos/traffic_video.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

device = "cpu"
model = models.get('yolo_nas_s', num_classes=1, checkpoint_path="./weights/ckpt_best.pth").to(device)

class_names = ['license-plate']
# output = cv2.VideoWriter('./Video/Output.avi', cv2.VideoWriter_fourcc('M', 'J', 'p', 'G'), 10, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if ret:
        result = list(model.predict(frame, conf=0.4))[0]
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences = result.prediction.confidence
        labels = result.prediction.labels.tolist()
        for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
            bbox = np.array(bbox_xyxy)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = class_names[classname]
            conf = math.ceil(confidence*100)/100
            label = f'{class_name}{conf}'
            text = ocr_image(frame, x1, y1, x2, y2)
            label = text
            print(x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            background = x1 + text_size[0], y1 - text_size[1] - 3
            cv2.rectangle(frame, (x1, y1), background, [255, 144, 30], -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        # output.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()