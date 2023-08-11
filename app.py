import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
from license_plate_detection import *


def main():
    st.title("License Plate Detection and OCR with YOLO-NAS")
    app_mode = st.sidebar.selectbox(
            'Choose Detection Mode', ['Run on Image', 'Run on Video']
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

    elif app_mode == "Run on Video":
        confidence = st.sidebar.slider(
            'Confidece', min_value=0.0, max_value=1.0
        )
        video_file = st.sidebar.file_uploader(
            "Upload a Video", type=["mp4", "avi", "mov", "asf"]
        )
        demo = "./Videos/demo.mp4"
        tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        if not video_file:
            video_stream = cv2.VideoCapture(video_file)
            tffile.name = demo
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text("Input Video")
            st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file.read())
            vid = open(tffile.name, 'rb')
            vid_bytes = vid.read()
            st.sidebar.text("Input Video")
            st.sidebar.video(vid_bytes)

        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html=True)

        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**License Plate Number**")
            kpi2_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html=True)
        object_detction__ocr_video(tffile.name, confidence, kpi1_text, kpi2_text, stframe)




if __name__ == '__main__':
    main()