# Import necessary libraries
import cv2
from super_gradients.training import models
import numpy as np
import time
import math
import easyocr
import streamlit as st

# Initialize EasyOCR for text recognition
reader = easyocr.Reader(['en'], gpu=True)

# Define a function for OCR on a region of an image
def ocr_image(frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]  # Crop the frame to the specified region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert cropped region to grayscale
    result = reader.readtext(gray)  # Perform OCR on the grayscale region
    text = ""  # Initialize an empty string for detected text
    for res in result:
        if len(result) == 1:
            text = res[1]  # If only one result, set the text to the detected text
        if len(result) > 1 and len(res[1])>6 and res[2] >0.2:
            text = res[1]  # If multiple results and certain conditions met, set the text to the detected text
    return str(text)  # Return the detected text

# Define a function for object detection and OCR on a single image
def object_detection_ocr(image, confidence):
    device = "cpu"
    model = models.get('yolo_nas_s', 
                        num_classes=1, 
                        checkpoint_path='./Checkpoints/ckpt_best.pth').to(device)  # Load the object detection model
    class_names = ['license-plate']  # Define class names (in this case, just one class)

    result = list(model.predict(image, conf=confidence))[0]  # Perform object detection on the image
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()  # Extract bounding box coordinates
    confidence = result.prediction.confidence  # Extract confidence scores
    labels = result.prediction.labels.tolist()  # Extract class labels
    
    # Loop through detected objects
    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidence, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]  # Extract coordinates
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
        classname = int(cls)
        class_name = class_names[classname]  # Get the class name
        conf = math.ceil(confidence*100)/100  # Calculate rounded confidence score
        label = f"{class_name}{conf}"  # Create label for the object
        text = ocr_image(image, x1, y1, x2, y2)  # Perform OCR on the region of interest
        text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]  # Get size of label text
        background = x1 + text_size[0], y1 - text_size[1] - 3  # Calculate background size for label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Draw bounding box
        cv2.rectangle(image, (x1, y1), background, [255, 144, 20], -1, cv2.LINE_AA)  # Draw label background
        cv2.putText(image, text, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)  # Draw label text

    # Display the output image using Streamlit
    st.subheader('Output Image')
    st.image(image, channels="BGR", use_column_width=1)

# Define a function for object detection and OCR on a video
def object_detction__ocr_video(video, confidence, kpi1_text, kpi2_text, stframe):
    cap = cv2.VideoCapture(video)  # Initialize video capture from specified source

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    device = "cpu"
    model = models.get('yolo_nas_s', 
                        num_classes=1, 
                        checkpoint_path='./Checkpoints/ckpt_best.pth').to(device)  # Load the object detection model
    class_names = ['license-plate']  # Define class names (in this case, just one class)

    prev_time = 0
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if ret:
            result = list(model.predict(frame, conf=0.4))[0]  # Perform object detection on the frame
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()  # Extract bounding box coordinates
            confidences = result.prediction.confidence  # Extract confidence scores
            labels = result.prediction.labels.tolist()  # Extract class labels
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]  # Extract coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert coordinates to integers
                classname = int(cls)
                class_name = class_names[classname]  # Get the class name
                conf = math.ceil(confidence*100)/100  # Calculate rounded confidence score
                label = f'{class_name}{conf}'  # Create label for the object
                text = ocr_image(frame, x1, y1, x2, y2)  # Perform OCR on the region of interest
                label = text  # Set label as the detected text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Draw bounding box
                text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]  # Get size of label text
                background = x1 + text_size[0], y1 - text_size[1] - 3  # Calculate background size for label
                cv2.rectangle(frame, (x1, y1), background, [255, 144, 30], -1, cv2.LINE_AA)  # Draw label background
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)  #
