"""
This script processes an image of a QR code to detect specific dot patterns and adjust the image 
rotation to equalize distances between detected dots. It utilizes OpenCV for image processing, 
NumPy for numerical operations, and Matplotlib for visualization.

Classes:
    ImageProcessor
    RotatedImageProcessor

Functions:
    - preprocess_image
    - detect_dots
    - draw_vertical_line
    - get_vertical_line_dots
    - visualize_dots
    - calculate_distances
    - rotate_image
    - rotate_image_to_equalize_distances
    - rotate_image
    - process_rotated_image
    - get_vertical_line_dots_rotated
    - visualize_rotated_dots
    - process_and_visualize_rotations

Dependencies:
    - OpenCV: cv2
    - NumPy: np
    - Matplotlib: matplotlib.pyplot as plt
    - SciPy: minimize from scipy.optimize
    - random

Usage:
    The script can be executed as a standalone program. The main block demonstrates the
    usage of the `ImageProcessor` and `RotatedImageProcessor` classes to preprocess the image,
    detect dots, draw a vertical line, visualize dots, and rotate the image to equalize distances
    between detected dots.

Example:
    $ python main_3.py

    This example demonstrates how to use the `ImageProcessor` and `RotatedImageProcessor`
    classes to process an image, detect dots, and rotate the image for better alignment.

Classes:
    ImageProcessor:
        A class for processing an image to detect dots and analyze their positions relative
        to a vertical center line. It includes methods for preprocessing the image, detecting
        dots, drawing a vertical line, visualizing detected dots, calculating distances between
        dots, and rotating the image to equalize distances.

        Methods:
            - __init__(self, image_path)
            - preprocess_image(self)
            - detect_dots(self)
            - draw_vertical_line(self)
            - get_vertical_line_dots(self, threshold=5)
            - visualize_dots(self)
            - calculate_distances(self, vertical_line_dots, angle=0)
            - rotate_image(self, angle_degrees)
            - rotate_image_to_equalize_distances(self)

    RotatedImageProcessor:
        A subclass of ImageProcessor that additionally processes rotated images at various angles
        and visualizes the detected dots. It includes methods for rotating the image, processing
        rotated images, getting vertical line dots in rotated images, and visualizing rotated dots.

        Methods:
            - __init__(self, image_path)
            - rotate_image(self, angle)
            - process_rotated_image(self, angle)
            - get_vertical_line_dots_rotated(self, rotated_dots, rot_mat, threshold=5)
            - visualize_rotated_dots(self, rotated_image, rotated_dots, angle)
            - process_and_visualize_rotations(self)

Main:
    If this script is executed as the main module, it will create an instance of `ImageProcessor`,
    preprocess the image, detect dots, draw a vertical line, visualize dots, calculate distances,
    and attempt to rotate the image to equalize distances. Additionally, it will create an instance
    of `RotatedImageProcessor` to process and visualize rotations at various angles.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.dots = []
        self.vertical_center = self.image.shape[1] // 2
        self.center_point = (self.vertical_center, self.image.shape[0] // 2)
        self.image_with_line = None
        self.opened_image = None

    def preprocess_image(self):
        self.blurred_image = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.thresh_image = cv2.adaptiveThreshold(
            self.blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.opened_image = cv2.morphologyEx(self.thresh_image, cv2.MORPH_OPEN, kernel)

    def detect_dots(self):
        if self.opened_image is None:
            raise RuntimeError("Image has not been preprocessed. Call preprocess_image() first.")
        contours, _ = cv2.findContours(
            self.opened_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    self.dots.append((cX, cY))

    def draw_vertical_line(self):
        self.image_with_line = cv2.line(self.image.copy(), (self.vertical_center, 0), (self.vertical_center, self.image.shape[0]), (255, 0, 0), 1)
        self.vertical_line_coords = [(self.vertical_center, y) for y in range(self.image.shape[0])]

    def get_vertical_line_dots(self, threshold=5):
        vertical_line_dots = [
            (x, y) for x, y in self.dots if abs(x - self.vertical_center) <= threshold
        ]
        return vertical_line_dots

    def visualize_dots(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image_with_line, cmap='gray')
        if self.dots:
            vertical_line_dots = self.get_vertical_line_dots()
            if vertical_line_dots:
                plt.scatter(*zip(*vertical_line_dots), color='red', s=20, label='Detected Dots on Vertical Line')
        plt.scatter([self.center_point[0]], [self.center_point[1]], color='blue', s=100, label='Center Point')
        plt.legend()
        plt.title("Detected Dots and Center Line in Original Image")
        plt.show()
        
        print("\nDetected Dots Coordinates (Original Image):", vertical_line_dots)
        
        return vertical_line_dots

    def calculate_distances(self, vertical_line_dots, angle=0):
        if len(vertical_line_dots) < 6:
            return {"average_distance": np.inf, "slope": 0, "top_average_distance": 0, "bottom_average_distance": 0}

        vertical_line_dots.sort(key=lambda x: x[1])

        top_three = vertical_line_dots[:3]
        bottom_three = vertical_line_dots[-3:]

        top_distances = np.linalg.norm(np.diff(top_three, axis=0), axis=1)
        bottom_distances = np.linalg.norm(np.diff(bottom_three, axis=0), axis=1)

        top_average_distance = np.mean(top_distances)
        bottom_average_distance = np.mean(bottom_distances)
        
        avg_distance = (top_average_distance + bottom_average_distance) / 2

        if angle == 0:
            slope = 0
        else:
            x_coords = [x for x, _ in vertical_line_dots]
            y_coords = [y for _, y in vertical_line_dots]
            slope = np.polyfit(y_coords, x_coords, 1)[0]

        return {
            "average_distance": avg_distance,
            "slope": slope,
            "top_average_distance": top_average_distance,
            "bottom_average_distance": bottom_average_distance
        }

    def rotate_image(self, angle_degrees):
        angle_degrees = float(angle_degrees)  # Ensure angle is treated as a scalar
        height, width = self.image.shape
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        rotated_image = cv2.warpAffine(self.image_with_line, rotation_matrix, (width, height))
        return rotated_image

    def rotate_image_to_equalize_distances(self):
        def objective_function(angle):
            angle = angle[0]  # Extract scalar from array
            rotated_image = self.rotate_image(angle)
            rotated_processor = ImageProcessor(self.image_path)
            rotated_processor.image_with_line = rotated_image
            rotated_processor.preprocess_image()  # Ensure preprocessing
            rotated_processor.detect_dots()
            vertical_line_dots = rotated_processor.get_vertical_line_dots()
            distances = rotated_processor.calculate_distances(vertical_line_dots)
            return abs(distances["top_average_distance"] - distances["bottom_average_distance"])

        self.preprocess_image()
        self.detect_dots()
        self.draw_vertical_line()

        result = minimize(objective_function, x0=[0], bounds=[(-30, 30)], method='L-BFGS-B')
        optimal_angle = result.x[0]

        final_rotated_image = self.rotate_image(optimal_angle)
        self.image_with_line = final_rotated_image
        self.preprocess_image()  # Ensure preprocessing
        self.detect_dots()
        vertical_line_dots = self.get_vertical_line_dots()
        distances = self.calculate_distances(vertical_line_dots)
        # print(f"Optimal Angle: {optimal_angle:.2f} degrees")
        print(f"Average distance between top and bottom three dots is, ", distances["average_distance"])
        # print(f"Slope of the vertical line (Rotated by {optimal_angle:.2f} degrees):", distances["slope"])
        print("\n")

        return final_rotated_image, optimal_angle

class RotatedImageProcessor(ImageProcessor):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.angles = [random.randint(-10, 10) for _ in range(5)]

    def rotate_image(self, angle):
        image_center = tuple(np.array(self.image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rotated_image = cv2.warpAffine(self.image_with_line, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image, rot_mat

    def process_rotated_image(self, angle):
        self.preprocess_image()

        rotated_image, rot_mat = self.rotate_image(angle)
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

        return rotated_image, rotated_dots, rot_mat

    def get_vertical_line_dots_rotated(self, rotated_dots, rot_mat, threshold=5):
        vertical_center_coords = np.array(self.vertical_line_coords, dtype=np.float32)
        rotated_center_coords = cv2.transform(np.array([vertical_center_coords]), rot_mat)[0]
        
        vertical_line_dots_rotated = [
            (x, y) for x, y in rotated_dots if any(abs(x - int(rx)) <= threshold and abs(y - int(ry)) <= threshold for rx, ry in rotated_center_coords)
        ]
        return vertical_line_dots_rotated

    def visualize_rotated_dots(self, rotated_image, rotated_dots, angle):
        plt.figure(figsize=(10, 10))
        plt.imshow(rotated_image, cmap='gray')
        if rotated_dots:
            plt.scatter(*zip(*rotated_dots), color='red', s=20, label=f'Detected Dots (Rotated by {angle} degrees)')
        plt.legend()
        plt.title(f"Detected Dots in Image Rotated by {angle} degrees")
        plt.show()

    def process_and_visualize_rotations(self):
        for angle in self.angles:
            rotated_image, rotated_dots, rot_mat = self.process_rotated_image(angle)
            vertical_line_dots_rotated = self.get_vertical_line_dots_rotated(rotated_dots, rot_mat)
            
            distances = self.calculate_distances(vertical_line_dots_rotated, angle)
            print(f"Average distance between top and bottom three dots (Rotated by {angle} degrees):", distances["average_distance"])
            print(f"Slope of the vertical line (Rotated by {angle} degrees):", distances["slope"])
            print("\n")

            self.visualize_rotated_dots(rotated_image, vertical_line_dots_rotated, angle)


# Main entry point of the SQCode processing
if __name__ == "__main__":
    processor = ImageProcessor(image_path="./SQcode-5.png")
    processor.preprocess_image()
    processor.detect_dots()
    processor.draw_vertical_line()
    vertical_line_dots = processor.visualize_dots()
    
    print("Vertical line dots in original image:", vertical_line_dots)
    distances = processor.calculate_distances(vertical_line_dots)
    print("Average distance between top and bottom three dots:", distances["average_distance"])
    print("Slope of the vertical line:", distances["slope"])
    print("\n")
    
    rotated_processor = RotatedImageProcessor(image_path="./SQcode-5.png")
    rotated_processor.draw_vertical_line()
    rotated_processor.process_and_visualize_rotations()

    rotated_image, final_angle = processor.rotate_image_to_equalize_distances()
    
    # print(rotated_image, final_angle)
    plt.figure(figsize=(10, 10))
    plt.imshow(rotated_image, cmap='gray')
    plt.title(f"Image Rotated to Equalize Distances (Angle: {final_angle} degrees)")
    plt.show()
