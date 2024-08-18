"""
Image Processing and Dot Detection Module

This module provides functionality for processing grayscale images to detect dot-like features.
It includes preprocessing steps such as Gaussian blurring, adaptive thresholding, and morphological
operations. The module also allows for image rotation and subsequent re-detection of dots to assess
the stability of detection under small perturbations.

Classes:
    ImageProcessor: Handles image loading, preprocessing, and dot detection.
    RotatedImageProcessor: Inherits from ImageProcessor and adds rotation functionality.

Functions:
    rotate_image(image, angle): Rotates an image by a specified angle.

Usage Example:
    # Initialize the image processor with the path to the image
    processor = ImageProcessor(image_path="./SQcode-5.png")

    # Preprocess the image and detect dots
    processor.preprocess_image()
    processor.detect_dots()

    # Visualize detected dots on the original image
    processor.visualize_dots()

    # Initialize the rotated image processor for random rotations
    rotated_processor = RotatedImageProcessor(image_path="./SQcode-5.png")
    rotated_processor.process_and_visualize_rotations()
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

class ImageProcessor:
    def __init__(self, image_path):
        """
        Initializes the ImageProcessor with the provided image path.
        
        Args:
            image_path (str): Path to the image file.
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.dots = []
        self.vertical_center = self.image.shape[1] // 2
        self.center_point = (self.vertical_center, self.image.shape[0] // 2)

    def preprocess_image(self):
        """
        Preprocesses the image by applying Gaussian blur, adaptive thresholding, and morphological opening.
        """
        self.blurred_image = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.thresh_image = cv2.adaptiveThreshold(
            self.blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.opened_image = cv2.morphologyEx(self.thresh_image, cv2.MORPH_OPEN, kernel)

    def detect_dots(self):
        """
        Detects dots in the preprocessed image by finding and filtering contours based on area.
        """
        contours, _ = cv2.findContours(
            self.opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:  # Adjust the threshold based on dot size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    self.dots.append((cX, cY))

    def visualize_dots(self):
        """
        Visualizes the detected dots on the original image.
        """
        # Draw the vertical line on the image
        image_with_line = cv2.line(self.image.copy(), (self.vertical_center, 0), (self.vertical_center, self.image.shape[0]), (255, 0, 0), 1)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_line, cmap='gray')
        if self.dots:
            plt.scatter(*zip(*self.dots), color='red', s=20, label='Detected Dots')
        plt.scatter([self.center_point[0]], [self.center_point[1]], color='blue', s=100, label='Center Point')
        plt.legend()
        plt.title("Detected Dots and Center Line in Original Image")
        plt.show()

class RotatedImageProcessor(ImageProcessor):
    def __init__(self, image_path):
        """
        Initializes the RotatedImageProcessor with the provided image path.
        
        Args:
            image_path (str): Path to the image file.
        """
        super().__init__(image_path)
        self.angles = [random.randint(-10, 10) for _ in range(5)]

    def rotate_image(self, angle):
        """
        Rotates the image by a specified angle.
        
        Args:
            angle (float): Angle by which to rotate the image.

        Returns:
            np.ndarray: The rotated image.
        """
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated_image = cv2.warpAffine(self.image, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image

    def process_rotated_image(self, angle):
        """
        Processes a rotated image by applying the same preprocessing and dot detection steps.
        
        Args:
            angle (float): Angle by which to rotate the image.
        """
        rotated_image = self.rotate_image(angle)
        blurred = cv2.GaussianBlur(rotated_image, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rotated_dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    rotated_dots.append((cX, cY))

        return rotated_image, rotated_dots

    def visualize_rotated_dots(self, rotated_image, rotated_dots, angle):
        """
        Visualizes the detected dots on a rotated image.
        
        Args:
            rotated_image (np.ndarray): The rotated image.
            rotated_dots (list of tuple): List of detected dot coordinates in the rotated image.
            angle (float): The angle by which the image was rotated.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(rotated_image, cmap='gray')
        if rotated_dots:
            plt.scatter(*zip(*rotated_dots), color='red', s=20, label=f'Detected Dots (Rotated by {angle} degrees)')
        plt.legend()
        plt.title(f"Detected Dots in Image Rotated by {angle} degrees")
        plt.show()

    def process_and_visualize_rotations(self):
        """
        Processes and visualizes the image for a series of random rotations.
        """
        for angle in self.angles:
            rotated_image, rotated_dots = self.process_rotated_image(angle)
            print(f"\nLength of rotated dots (Rotated by {angle} degrees): ", len(rotated_dots))
            print(f"Rotated dots coordinates (Rotated by {angle} degrees):", rotated_dots)
            self.visualize_rotated_dots(rotated_image, rotated_dots, angle)

# main entrypoint of the SQCode processing
if __name__ == "__main__":
    processor = ImageProcessor(image_path="./SQcode-5.png")
    processor.preprocess_image()
    processor.detect_dots()
    processor.visualize_dots()

    rotated_processor = RotatedImageProcessor(image_path="./SQcode-5.png")
    rotated_processor.process_and_visualize_rotations()
