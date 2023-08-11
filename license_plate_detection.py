import cv2
from super_gradients.training import models
import numpy as np
import time
import math
import easyocr
import streamlit as st

reader = easyocr.Reader(['en'], gpu=True)
def ocr_image(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    text = ""
    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1])>6 and res[2] >0.2:
            text = res[1]
    return str(text)


def object_detection_ocr(image, confidence):
    device = "cpu"
    model = models.get('yolo_nas_s', 
                        num_classes=1, 
                        checkpoint_path='./Checkpoints/ckpt_best.pth').to(device)
    class_names = ['license-plate']

    result = list(model.predict(image, conf=confidence))[0]
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
    confidence = result.prediction.confidence
    labels = result.prediction.labels.tolist()
    
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidence, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        classname = int(cls)
        class_name = class_names[classname]
        conf = math.ceil(confidence*100)/100
        label = f"{class_name}{conf}"
        text = ocr_image(image, x1, y1, x2, y2)
        text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        background = x1 + text_size[0], y1 - text_size[1] - 3       
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.rectangle(image, (x1, y1), background, [255, 144, 20], -1, cv2.LINE_AA)
        cv2.putText(image, text, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    st.subheader('Output Image')
    st.image(image, channels="BGR", use_column_width=1)

def object_detction__ocr_video(video, confidence, kpi1_text, kpi2_text, stframe):
    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    device = "cpu"
    model = models.get('yolo_nas_s', 
                        num_classes=1, 
                        checkpoint_path='./Checkpoints/ckpt_best.pth').to(device)
    class_names = ['license-plate']

    prev_time = 0
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                background = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), background, [255, 144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            stframe.image(frame, channels='BGR', use_column_width=True)
            current_time = time.time()
            fps = 1/(current_time - prev_time)
            prev_time = current_time

            kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)} FPS</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.7s}'.format(label)}</h1>", unsafe_allow_html=True)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()