import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from license_plate_detection import *


def main():
    st.title("License Plate Detection and OCR with YOLO-NAS")
    app_mode = st.sidebar.selectbox(
            'Choose Detection Mode', ['Run on Image', 'Run on Vide']
    )
    if app_mode == 'Run on Image':
        confidence = st.sidebar.slider(
            'Confidece', min_value=0.0, max_value=1.0
        )
        img_file = st.sidebar.file_uploader(
            "upload an Image", type=["png", "jpg", "jpeg"]
        )
        demo = "./OCR-2/valid/images/195_jpg.rf.609d4d0dfd006181eb7d52a90c00bfd8.jpg"
        if img_file is not None:
            img = cv2.imdecode(np.fromstring(
                img_file.read(), np.uint8), 1
            )
            image = np.array(Image.open(img_file))
        else:
            img = cv2.imread(demo)
            image = np.array(Image.open(demo))
        st.sidebar.text("Original Image")
        st.sidebar.image(image)
        object_detection_ocr(img, confidence)
if __name__ == '__main__':
    main()