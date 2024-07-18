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
        Preprocesses the image by applying Gaussian blur, adaptive threshold, and morphological opening.
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

    def draw_vertical_line(self):
        """
        Draws a vertical line at the center of the image.
        """
        self.image_with_line = cv2.line(self.image.copy(), (self.vertical_center, 0), (self.vertical_center, self.image.shape[0]), (255, 0, 0), 1)
        self.vertical_line_coords = [(self.vertical_center, y) for y in range(self.image.shape[0])]

    def get_vertical_line_dots(self, threshold=5):
        """
        Filters the detected dots to get those along the vertical line at the image's center.

        Args:
            threshold (int): Maximum distance from the vertical line to consider a dot as being on the line.
        
        Returns:
            list of tuple: List of coordinates of dots along the vertical line.
        """
        vertical_line_dots = [
            (x, y) for x, y in self.dots if abs(x - self.vertical_center) <= threshold
        ]
        return vertical_line_dots

    def visualize_dots(self):
        """
        Visualizes the detected dots on the original image.
        """
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
        
        # Print all the coordinates of the dots detected in the original image
        print("\nDetected Dots Coordinates (Original Image):", vertical_line_dots)
        
        return vertical_line_dots

    def calculate_distances(self, vertical_line_dots, angle=0):
        """
        Calculate distances between dots and other relevant metrics.
        
        Args:
            vertical_line_dots (list of tuple): List of coordinates of dots along the vertical line.
            angle (float): Angle by which the image was rotated.
        
        Returns:
            dict: Dictionary with the calculated distances and slope.
        """
        if len(vertical_line_dots) < 6:
            raise ValueError("Not enough dots to perform the calculations. At least 6 dots are required.")

        vertical_line_dots.sort(key=lambda x: x[1])  # Sort dots by their y-coordinate (vertical position)

        top_three = vertical_line_dots[:3]
        bottom_three = vertical_line_dots[-3:]
        
        # Calculate the average distance using both x and y coordinates
        top_distances = np.linalg.norm(np.diff(top_three, axis=0), axis=1)
        bottom_distances = np.linalg.norm(np.diff(bottom_three, axis=0), axis=1)

        top_average_distance = np.mean(top_distances)
        bottom_average_distance = np.mean(bottom_distances)
        
        avg_distance = (top_average_distance + bottom_average_distance) / 2

        # Calculate the slope of the vertical line
        if angle == 0:
            slope = 0  # Original image (no rotation)
        else:
            x_coords = [x for x, _ in vertical_line_dots]
            y_coords = [y for _, y in vertical_line_dots]
            slope = np.polyfit(y_coords, x_coords, 1)[0]

        return {
            "average_distance": avg_distance,
            "slope": slope
        }

class RotatedImageProcessor(ImageProcessor):
    def __init__(self, image_path):
        """
        Initializes the RotatedImageProcessor with the provided image path and 
        create random angle within the range of the  -10, 10.
        
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
        rotated_image = cv2.warpAffine(self.image_with_line, rot_mat, self.image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated_image, rot_mat

    def process_rotated_image(self, angle):
        """
        Processes a rotated image by applying the same preprocessing and dot detection steps.
        
        Args:
            angle (float): Angle by which to rotate the image.
        """
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
        """
        Filters the detected dots to get those along the vertical line in the rotated image.
        
        Args:
            rotated_dots (list of tuple): List of detected dot coordinates in the rotated image.
            rot_mat (np.ndarray): Rotation matrix used to rotate the image.
            threshold (int): Maximum distance from the vertical line to consider a dot as being on the line.
        
        Returns:
            list of tuple: List of coordinates of dots along the vertical line in the rotated image.
        """
        # Compute the new vertical center after rotation
        vertical_center_coords = np.array(self.vertical_line_coords, dtype=np.float32)
        rotated_center_coords = cv2.transform(np.array([vertical_center_coords]), rot_mat)[0]
        
        vertical_line_dots_rotated = [
            (x, y) for x, y in rotated_dots if any(abs(x - int(rx)) <= threshold and abs(y - int(ry)) <= threshold for rx, ry in rotated_center_coords)
        ]
        return vertical_line_dots_rotated

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
            rotated_image, rotated_dots, rot_mat = self.process_rotated_image(angle)
            vertical_line_dots_rotated = self.get_vertical_line_dots_rotated(rotated_dots, rot_mat)
            
            # Calculate distances for rotated images
            distances = self.calculate_distances(vertical_line_dots_rotated, angle)
            print(f"Average distance between top and bottom three dots (Rotated by {angle} degrees):", distances["average_distance"])
            print(f"Slope of the vertical line (Rotated by {angle} degrees):", distances["slope"])
            print("\n")

            
            # Visualize rotated dots
            self.visualize_rotated_dots(rotated_image, vertical_line_dots_rotated, angle)

# Main entry point of the SQCode processing
if __name__ == "__main__":
    processor = ImageProcessor(image_path="./SQcode-5.png")
    processor.preprocess_image()
    processor.detect_dots()
    processor.draw_vertical_line()
    vertical_line_dots = processor.visualize_dots()
    
    print("Vertical line dots in original image:", vertical_line_dots)
    # Calculate distances and slopes for the original image
    distances = processor.calculate_distances(vertical_line_dots)
    print("Average distance between top and bottom three dots:", distances["average_distance"])
    print("Slope of the vertical line:", distances["slope"])
    print("\n")
    
    # Process and visualize rotations
    rotated_processor = RotatedImageProcessor(image_path="./SQcode-5.png")
    rotated_processor.draw_vertical_line()  # Ensure the vertical line is drawn in the rotated processor
    rotated_processor.process_and_visualize_rotations()
