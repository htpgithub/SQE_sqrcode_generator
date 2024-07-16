# Image Processing with Dot Detection and Rotation Visualization for SQRCode

This project processes an image to detect dots, visualizes the detected dots, and applies rotations to the image to visualize how dot detection changes with rotation.

## Features

- **Image Preprocessing**: Gaussian blur, adaptive thresholds, and morphological opening.
- **Dot Detection**: Contour detection and centroid calculation.
- **Visualization**: Plots the original and rotated images with detected dots.
- **Rotation Handling**: Processes and visualizes the image at various random rotations.

## Requirements

- numpy==1.26.4
- matplotlib==3.9.1
- opencv-python==4.10.0.84
- qrcode==7.4.2

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create a Virtual Environment

```bash
# For Unix or MacOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Program

Ensure you have an image named `SQcode-5.png` in the same directory as the script, or modify the `image_path` variable in the script to point to your image.

```bash
python script.py
```