import cv2
import qrcode
import numpy as np
import matplotlib.pyplot as plt
import json

"""Variables Start"""
black_color = "Black"
white_color = "White"
"""Variables End"""

"""Functions Start"""


def area_of_triangle(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def mirror_coordinates(x, y, center_x, center_y):
    """Mirror coordinates across both x and y axes relative to the center point."""
    mirrored_x = 2 * center_x - x
    mirrored_y = 2 * center_y - y
    return mirrored_x, mirrored_y


def char_to_custom_bits(char):
    """Convert character to a custom binary string with MSB set to '1'."""
    binary_string = format(ord(char), "08b")
    return binary_string


def text_to_bits_and_json(text):
    """Converts text to a binary string with custom MSB and returns JSON with individual characters and their bits."""
    binary_string = ""
    binary_dict = {}

    for char in text:
        custom_bits = char_to_custom_bits(char)
        binary_string += custom_bits
        binary_dict[char] = custom_bits

    # Convert the dictionary to JSON
    json_filename = "binary_data.json"
    with open(json_filename, "w") as json_file:
        json.dump(binary_dict, json_file, indent=4)

    print(f"Binary data saved to {json_filename}")
    return binary_string, binary_dict


def calculate_odd_row_positions(start, end, spacing):
    row_positions = []
    mid = (start + end) // 2
    row_positions.append(mid)
    for i in range(1, (end - start) // (2 * spacing) + 1):
        row_positions.append(mid - i * spacing)
        row_positions.append(mid + i * spacing)
    return sorted(row_positions)


def inside_square(x, y, image_size, square_side):
    """Check if a point (x, y) is inside the central square."""
    start_x = (image_size - square_side) // 2
    start_y = (image_size - square_side) // 2
    end_x = start_x + square_side
    end_y = start_y + square_side

    return start_x < x < end_x and start_y < y < end_y


"""Functions End"""


class DotPlacer:
    def __init__(self, x, y, processed_image, image_size, hexagon_radius, dot_spacing=35):
        self.x = x
        self.y = y
        self.inside_hexagon = False
        self.inside_QRCode = False
        self.center_dot = False
        self.color = black_color
        self.dot_spacing = dot_spacing
        self.dot_diameter = 0.6 * dot_spacing  # Dot diameter is 90% of the dot-to-dot dimension
        self.processed_image = processed_image
        self.image_size = image_size
        self.hexagon_radius = hexagon_radius

    def center_dot(self):
        return self.center_dot

    def show(self):
        dot_radius = int(self.dot_diameter // 3.5)
        cv2.circle(self.processed_image, (self.x, self.y), dot_radius, (0, 0, 255), -1)  # Red filled circle


class SQEDotPatternCode:
    def __init__(self, data, hexagon_radius=500, dot_spacing=10):
        self.data = data
        self.hexagon_radius = hexagon_radius
        self.dot_spacing = dot_spacing
        self.dot_diameter = 0.6 * dot_spacing  # Dot diameter is 90% of the dot-to-dot dimension
        self.image_size = 2 * hexagon_radius
        self.processed_image = None
        self.bits, self.binary_json = text_to_bits_and_json(data)
        self.square_side = int(self.hexagon_radius * 0.8)

    def create_image_canvas(self):
        """Create a blank image canvas where the hexagonal pattern will be added."""
        self.processed_image = (
                np.ones((self.image_size, self.image_size), dtype=np.uint8)
                * 255
        )

    def draw_hexagon(self):
        """Draw a hexagon shape on the image canvas with a tip pointing upwards."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        size = self.hexagon_radius

        vertices = np.array(
            [
                (
                    center_x + size * np.cos(np.pi / 6 + np.pi / 3 * i),
                    center_y + size * np.sin(np.pi / 6 + np.pi / 3 * i),
                )
                for i in range(6)
            ],
            np.int32,
        )
        cv2.polylines(self.processed_image, [vertices], isClosed=True, color=0, thickness=2)

    def add_qr_code(self):
        """Overlay the QR code at the center of the hexagon."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2

        qr_text = self.data
        qr = qrcode.QRCode(border=2)
        qr.add_data(qr_text)
        qr.make(fit=True)

        qr_image = qr.make_image(fill='black', back_color='white').convert('L')
        qr_image = np.array(qr_image)

        qr_image_resized = cv2.resize(qr_image, (self.square_side, self.square_side), interpolation=cv2.INTER_AREA)

        # Overlay QR code at the center of the hexagon
        qr_top_left_x = center_x - (qr_image_resized.shape[0] // 2)
        qr_top_left_y = center_y - (qr_image_resized.shape[1] // 2)

        for i in range(qr_image_resized.shape[0]):
            for j in range(qr_image_resized.shape[1]):
                if qr_image_resized[i, j] == 0:  # If the QR code pixel is black
                    x = qr_top_left_x + i
                    y = qr_top_left_y + j
                    self.processed_image[x, y] = 0  # Place the QR code pixel on the image

    def create_dot(self):
        cv_size = (2000, 2000)
        spacing = 18
        dot_number = 0

        # Loop through rows and columns to create a grid of circles
        for y in range(0, cv_size[1], spacing):
            for x in range(0, cv_size[0], spacing):
                dot = DotPlacer(x, y, self.processed_image, self.image_size, self.hexagon_radius)
                if self.inside_hexagon(x, y) and not inside_square(x, y, self.image_size, self.square_side):
                    dot.show()
                    dot_number += 1

    def inside_hexagon(self, x, y):
        """Check if a point (x, y) is inside the hexagon."""
        center_x = self.image_size // 2
        x1, y1 = (center_x, 10)
        x2, y2 = (int(center_x - (np.sin(45) * self.hexagon_radius)), int(np.cos(45) * self.hexagon_radius))
        x3, y3 = (int(center_x + (np.sin(45) * self.hexagon_radius)), int(np.cos(45) * self.hexagon_radius))
        start_qr_x = (self.image_size - self.square_side) // 2
        start_qr_y = (self.image_size - self.square_side) // 2
        end_qr_x = start_qr_x + self.square_side
        end_qr_y = start_qr_y + self.square_side
        # Calculate the area of the full triangle
        A = area_of_triangle(x1, y1, x2, y2, x3, y3)
        # Calculate the areas of the triangles formed with the point
        A1 = area_of_triangle(x, y, x2, y2, x3, y3)
        A2 = area_of_triangle(x1, y1, x, y, x3, y3)
        A3 = area_of_triangle(x1, y1, x2, y2, x, y)

        if y <= np.cos(45) * self.hexagon_radius:  # triangle
            return A == A1 + A2 + A3
        elif center_x - (np.sin(45) * self.hexagon_radius) < x < center_x + (
                np.sin(45) * self.hexagon_radius) and np.cos(
            45) * self.hexagon_radius < y < 323:
            return True
        elif end_qr_x < x < center_x + (np.sin(45) * self.hexagon_radius) and 323 < y < end_qr_y:  # rectangle
            return True

    def process_and_visualize(self):
        """Generate the hexagon with dots and visualize the result."""
        self.create_image_canvas()
        self.draw_hexagon()
        self.add_qr_code()
        self.create_dot()

        # Display the final image
        plt.imshow(self.processed_image, cmap="gray")
        plt.axis("off")
        plt.show()


# Main entry point for generating the pattern
if __name__ == "__main__":
    processor = SQEDotPatternCode(data="https://organicminded.org/questionAnswers")
    processor.process_and_visualize()
