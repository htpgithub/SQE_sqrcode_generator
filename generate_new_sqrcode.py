import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize


class ImageProcessor:
    def __init__(self, image_path, message_length, error_correction_level):
        self.image_path = image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.processed_image = None
        self.dots = None
        self.vertical_line_dots = None
        self.dot_distance = None
        self.message_length = message_length
        self.error_correction_level = error_correction_level
        self.qr_code_size = None

    def preprocess_image(self):
        """Preprocess the image for dot detection."""
        _, self.processed_image = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY)

    def detect_dots(self):
        """Detect dots in the preprocessed image."""
        contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.dots = [cv2.boundingRect(contour) for contour in contours]

    def draw_vertical_line(self):
        """Draw a vertical line following the client's specifications."""
        height, width = self.processed_image.shape
        vertical_center = width // 2
        line_length = height
        self.vertical_line_dots = [(vertical_center, i) for i in range(line_length)]
        
        # Ensure the first bit is always 1 and follow the alternating pattern
        for i in range(9):
            self.processed_image[i, vertical_center] = 255 if i % 2 == 0 else 0

        cv2.line(self.processed_image, (vertical_center, 0), (vertical_center, height), 255, 1)

    def get_vertical_line_dots(self, threshold=5):
        """Extract the dots near the vertical line."""
        self.vertical_line_dots = [(x, y) for (x, y, w, h) in self.dots if abs(x - self.processed_image.shape[1] // 2) <= threshold]
        return self.vertical_line_dots

    def visualize_dots(self):
        """Visualize the detected dots."""
        output_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in self.dots:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(output_image)
        plt.show()

    def calculate_distances(self):
        """Calculate the distances between the vertical line dots."""
        if len(self.vertical_line_dots) > 1:
            self.dot_distance = np.mean([self.vertical_line_dots[i + 1][1] - self.vertical_line_dots[i][1] for i in range(len(self.vertical_line_dots) - 1)])
        else:
            self.dot_distance = None
        return self.dot_distance

    def rotate_image(self, angle):
        """Rotate the image by a given angle."""
        (h, w) = self.processed_image.shape[:2]
        (cx, cy) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cx, cy), float(angle), 1.0)
        rotated_image = cv2.warpAffine(self.processed_image, M, (w, h))
        return rotated_image

    def rotate_image_to_equalize_distances(self):
        """Rotate the image to equalize the distances between vertical line dots."""
        result = minimize(lambda angle: abs(self.calculate_distances() - self.calculate_distances()), 0)
        optimal_angle = result.x[0]  # Extracting the angle value properly
        self.processed_image = self.rotate_image(optimal_angle)

    def process_and_visualize_rotations(self):
        """Process and visualize image rotations."""
        self.preprocess_image()
        self.detect_dots()
        self.draw_vertical_line()
        self.get_vertical_line_dots()
        self.visualize_dots()
        if self.dot_distance is not None:
            self.rotate_image_to_equalize_distances()
            self.visualize_dots()

    def set_dot_diameter(self):
        """Set the dot diameter as 0.9 times the distance between dots."""
        if self.dot_distance is None:
            self.calculate_distances()
        dot_diameter = 0.9 * self.dot_distance if self.dot_distance is not None else None
        return dot_diameter

    def calculate_qr_code_size(self):
        """Calculate the size of the embedded QR code based on the message length and error correction level."""
        base_size = 21  # Base size for version 1 QR Code
        additional_size = 4 * (self.message_length // 20)  # Simplified logic for increasing size
        error_correction_factor = 1 + self.error_correction_level * 0.1  # Factor based on error correction level
        self.qr_code_size = base_size + additional_size * error_correction_factor
        return self.qr_code_size

    def ensure_consistent_dot_placement(self):
        """Ensure that the dot distances are equal both vertically and horizontally."""
        if self.dot_distance is None:
            self.calculate_distances()
        
        # Re-calculate dot positions to ensure consistency
        if self.dot_distance is not None:
            new_dots = []
            for (x, y, w, h) in self.dots:
                new_x = round(x / self.dot_distance) * self.dot_distance
                new_y = round(y / self.dot_distance) * self.dot_distance
                new_dots.append((new_x, new_y, w, h))
            self.dots = new_dots

# Running the updated processing on the provided image
# image_processor = ImageProcessor('/mnt/data/SQcode-5.png', message_length=100, error_correction_level=2)
# image_processor.process_and_visualize_rotations()
# dot_diameter = image_processor.set_dot_diameter()
# qr_code_size = image_processor.calculate_qr_code_size()
# image_processor.ensure_consistent_dot_placement()

# dot_diameter, qr_code_size


class RotatedImageProcessor(ImageProcessor):
    def __init__(self, image_path, message_length, error_correction_level):
        super().__init__(image_path, message_length, error_correction_level)
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
    processor = ImageProcessor(image_path="./SQcode-5.png",  message_length=100, error_correction_level=2 )
    processor.preprocess_image()
    processor.detect_dots()
    processor.draw_vertical_line()
    vertical_line_dots = processor.visualize_dots()
    
    print("Vertical line dots in original image:", vertical_line_dots)
    # distances = processor.calculate_distances(vertical_line_dots)
    # print("Average distance between top and bottom three dots:", distances["average_distance"])
    # print("Slope of the vertical line:", distances["slope"])
    print("\n")
    
    rotated_processor = RotatedImageProcessor(image_path="./SQcode-5.png", message_length=100, error_correction_level=2)
    rotated_processor.draw_vertical_line()
    rotated_processor.process_and_visualize_rotations()

    rotated_image, final_angle = processor.rotate_image_to_equalize_distances()
    
    # print(rotated_image, final_angle)
    plt.figure(figsize=(10, 10))
    plt.imshow(rotated_image, cmap='gray')
    plt.title(f"Image Rotated to Equalize Distances (Angle: {final_angle} degrees)")
    plt.show()
