# Automatic Number Plate Recognition using YOLO-NAS and EasyOCR (Images & Videos)

<img src="./Videos/streamlit_demo.gif">

This project uses YOLO-NAS and EasyOCR to detect license plates and perform Optical Character Recognition (OCR) on them. The project includes both image and video processing capabilities, and has been deployed as a Streamlit web application. This is an update to a previous project, [Optical-Character-Recognition-WebApp](https://github.com/JacobJ215/Optical-Character-Recognition-WebApp)

[GitHub](https://github.com/JacobJ215/YOLO-NAS-OCR-WebApp)

## Features
* Real-time license plate detection using YOLO-NAS
* Optical Character Recognition (OCR) using EasyOCR
* Interactive user interface built with Streamlit

## Dataset
The dataset used for training and testing the YOLO-NAS model contains 484 annotated images of cars with license plates. The images were sourced from "Brave Images", "Google Images", and "https://flickr.com/". The annotations were made using [RoboFlow](https://app.roboflow.com/yolotraining-dfaoh/ocr-nsde5/deploy/1). 


## Project Overview
This project builds upon an [earlier version](https://github.com/JacobJ215/Optical-Character-Recognition-WebApp) that used YOLO-v5 and InceptionResNetV2 . The major changes and updates in this version include:
* Transition from YOLO-v5 to YOLO-NAS for license plate detection
* Replacement of pytesseract with EasyOCR for more accurate text extraction
* Training the YOLO-NAS model for 15 epochs using Google Colab
* Deployment as a Streamlit web application

## Evaluation

* 'PPYoloELoss/loss_cls': 0.9181855
* 'PPYoloELoss/loss_iou': 0.17447565
* 'PPYoloELoss/loss_dfl': 0.9500977
* 'PPYoloELoss/loss': 1.8294234

* 'Precision@0.50': 0.02825159952044487
* 'Recall@0.50': 1.0
* 'mAP@0.50': 0.9623407125473022
* 'F1@0.50': 0.05495075136423111

## Running the Web Application

1. Clone the repository:
```
git clone https://github.com/JacobJ215/YOLO-NAS-OCR-WebApp/tree/main"
cd YOLO-NAS-OCR-WebApp
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run app.py
```

## Usage
The web application provides two main modes:
Run on Image: Upload an image containing a license plate, and the application will perform real-time detection and OCR, displaying the extracted text from the license plate.
### Run on Image
![](Screenshots/run_on_image.jpg)

Run on Video: Upload a video with license plates, and the application will perform real-time detection and OCR on the video frames, showing the extracted text and the 
frame rate.

### Run on Video
![](Screenshots/run_on_video.png)



## Acknowledgments
Inspired by https://github.com/MuhammadMoinFaisal

The YOLO-NAS model used in this project is based on Super-Gradients Repository.

EasyOCR is an excellent OCR library developed by Jaided AI.
