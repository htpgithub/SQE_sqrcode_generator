# Image Processing with Dot Detection and Rotation Visualization for SQRCode

This project processes an image to detect dots, visualizes the detected dots, and applies rotations to the image to visualize how dot detection changes with rotation. Additionally, it includes a module that generates a hexagonal dot pattern based on binary data derived from a text input, visualizing the pattern, and extracting coordinates of the black dots.

## Features

- **Image Preprocessing**: Gaussian blur, adaptive thresholds, and morphological opening.
- **Dot Detection**: Contour detection and centroid calculation.
- **Visualization**: Plots the original and rotated images with detected dots.
- **Rotation Handling**: Processes and visualizes the image at various random rotations.
- **Hexagon Dot Pattern Generation**: Creates a hexagonal dot pattern based on binary data from text input, saving the binary data and dot coordinates to JSON files.

## Requirements

- numpy==1.26.4
- matplotlib==3.9.1
- opencv-python==4.10.0.84
- qrcode==7.4.2
- reedsolo==1.5.5

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

### 4. Run the Dot Detection and Rotation Program

Ensure you have an image named `SQcode-5.png` in the same directory as the script, or modify the `image_path` variable in the script to point to your image.

```bash
python script.py
```

### 5. Run the Hexagon Dot Pattern Generator

You can generate a hexagonal dot pattern based on your input text by running the following command. The program will create and display the pattern, and save the binary data and dot coordinates to JSON files.

```bash
python hexagon_dot_pattern.py
```

Ensure you provide a string of text data in the script (e.g., "Welcome To SQE, The Quantum Secure Blockchain Platform"). The program will generate a corresponding hexagonal dot pattern and save the relevant binary data and dot coordinates to JSON files.

## Hexagon Dot Pattern Generator

This module generates a hexagonal dot pattern based on binary data derived from a text input. The pattern is visualized as an image where black dots represent binary '1' bits, and white spaces represent binary '0' bits. The dots are arranged within a hexagon, and their placement mirrors across all quadrants of the hexagon. Additionally, the program extracts and saves the coordinates of the black dots, as well as the binary data corresponding to the input text.

### Classes:
--------
**SQEDotPatternCode**:
- A class that handles the conversion of text to binary, creation of the hexagonal pattern, and visualization. The class also includes methods to extract the coordinates of the black dots representing binary '1' bits and save them to a JSON file.

### Methods:
--------
- **__init__(self, data, hexagon_radius=500, dot_diameter=25, dot_spacing=25)**: Initializes the SQEDotPatternCode class with input data, hexagon size parameters, and other configurations.

- **char_to_custom_bits(self, char)**: Converts a character to a custom binary string with the most significant bit (MSB) set to '1'.

- **text_to_bits_and_json(self, text)**: Converts the input text to a binary string using the custom MSB conversion, and saves the binary data to a JSON file.

- **create_image_canvas(self)**: Creates a blank image canvas where the hexagonal pattern will be drawn.

- **draw_hexagon(self)**: Draws a hexagon shape on the image canvas, oriented with one tip pointing upwards.

- **draw_square(self)**: Draws a central square inside the hexagon, which serves as a boundary where no dots will be placed.

- **place_dots(self)**: Places dots within the hexagon based on the binary data, and mirrors the dots across all four quadrants. Extracts and saves the coordinates of the black dots.

- **is_inside_square(self, x, y)**: Checks if a point (x, y) is inside the central square.

- **is_inside_hexagon(self, x, y)**: Checks if a point (x, y) is inside the hexagon.

- **process_and_visualize(self)**: Generates the hexagonal pattern with dots, extracts the coordinates of the black dots, saves them to a JSON file, and visualizes the final image.

### Usage:
------
To use this module, simply provide a string of text data, and the program will generate a corresponding hexagonal dot pattern and save the relevant binary data and dot coordinates to JSON files.

**Example**:
```python
processor = SQEDotPatternCode(
    data="Welcome To SQE, The Quantum Secure Blockchain Platform"
)
processor.process_and_visualize()
```

## Requirements:
-------------
- Python 3.x
- OpenCV (cv2)
- NumPy (np)
- Matplotlib (plt)
- Reed-Solomon Error Correction (reedsolo)
- JSON
```

This `README.md` provides a comprehensive overview of the project, including setup instructions, features, and usage examples for both the image processing with dot detection and the hexagon dot pattern generator.