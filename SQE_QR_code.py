import math

import cv2
import qrcode
import numpy as np
import matplotlib.pyplot as plt
import json

"""Variables Start"""
red_color = (0, 0, 252)
black_color = (0, 0, 0)
blue_color = (255, 0, 0)
size_arr: list = []
size_arr_index = -1
"""Variables End"""


def area_of_triangle(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def mirror_coordinates(x, y, center_x, center_y):
    """Mirror coordinates across both x and y axes relative to the center point."""
    mirrored_x = 2 * center_x - x
    mirrored_y = 2 * center_y - y + 50
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
    start_y = (image_size - square_side + 50) // 2
    end_x = start_x + square_side
    end_y = start_y + square_side

    return start_x < x < end_x and start_y < y < end_y


def includes_vertical_dots(x, center_x):
    return center_x - 10 >= x or x >= center_x + 10


def inside_hexagon(x, y, image_size):
    """Check if a point (x, y) is inside the hexagon."""
    center_x = image_size // 2
    center_y = (image_size + 50) // 2
    dy = abs(x - center_x) / image_size
    dx = abs(y - center_y) / image_size
    a = 0.25 * math.sqrt(3.0)

    return (dy <= a) and (a * dx + 0.25 * dy <= 0.5 * a)


"""Functions End"""


class DotPlacer:
    def __init__(self, x, y, processed_image, hexagon_radius, image_size, square_side):
        self.x = x
        self.y = y
        self.visible = True
        self.dot_diameter = int(4.5)
        self.color = black_color
        self.center = image_size // 2
        self.hexagon_radius = hexagon_radius
        self.processed_image = processed_image
        self.inside_hexagon = inside_hexagon(self.x, self.y, image_size)
        self.inside_QRCode = inside_square(self.x, self.y, image_size, square_side)
        self.vertical_dot = self.center - 10 <= self.x <= self.center + 10
        self.main_dot = False

    def set_vertical_dot(self, value: bool):
        self.vertical_dot = value

    def set_change_color(self, value: tuple):
        self.color = value

    def set_main_dot(self, value: bool):
        self.main_dot = value

    def show(self):
        global size_arr_index

        if self.vertical_dot and not self.main_dot:
            # Upper vertical Dots
            if self.y <= self.center:
                size_arr_index += 1
                if not size_arr[size_arr_index]: self.visible = False
            # Lower vertical Dots
            if self.y >= self.center:  # not show lower vertical dots
                if not size_arr[size_arr_index]: self.visible = False
                size_arr_index -= 1
                print(size_arr[size_arr_index], size_arr_index)

        if self.visible:
            cv2.circle(self.processed_image, (self.x, self.y), self.dot_diameter, self.color, -1)


class SQEDotPatternCode:
    def __init__(self, data, hexagon_radius=300, dot_spacing=10):
        self.count = 0
        self.data = data
        self.size = len(data)
        self.processed_image = None
        self.dot_spacing = dot_spacing
        self._dots: list[DotPlacer] = []
        self.vertical_dots: list[DotPlacer] = []
        self.image_size = 2 * hexagon_radius
        self.center = self.image_size // 2
        self.hexagon_radius = hexagon_radius
        self.dot_diameter = 0.6 * dot_spacing  # Dot diameter is 90% of the dot-to-dot dimension
        self.square_side = int(self.hexagon_radius * 0.8)
        self.bits, self.binary_json = text_to_bits_and_json(data)
        self.dot_number = sum(c.isalpha() for c in self.data) * 8
        j = 1
        for i in range(8):
            if (self.size & j) > 0:
                size_arr.append(True)
            else:
                size_arr.append(False)

            j = j + j

    def create_image_canvas(self):
        """Create a blank image canvas where the hexagonal pattern will be added."""
        self.processed_image = (np.ones((self.image_size + 50, self.image_size), dtype=np.uint8) * 255)

    def draw_hexagon(self):
        """Draw a hexagon shape on the image canvas with a tip pointing upwards."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        radius = self.hexagon_radius

        vertices = np.array(
            [
                (
                    center_x + radius * np.cos(np.pi / 6 + np.pi / 3 * i),
                    center_y + radius * np.sin(np.pi / 6 + np.pi / 3 * i),
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
        qr_top_left_x = center_x - (qr_image_resized.shape[0] // 2) + 30
        qr_top_left_y = center_y - (qr_image_resized.shape[1] // 2)

        for i in range(qr_image_resized.shape[0]):
            for j in range(qr_image_resized.shape[1]):
                if qr_image_resized[i, j] == 0:  # If the QR code pixel is black
                    x = qr_top_left_x + i
                    y = qr_top_left_y + j
                    self.processed_image[x, y] = 0  # Place the QR code pixel on the image

    def find_space(self, cv_size):
        dot_spacing = 9
        center_x = self.image_size // 2

        for space in range(dot_spacing, dot_spacing * 4):
            count = 0
            for y in range(0, cv_size[1], space):
                for x in range(0, cv_size[0], space):
                    if includes_vertical_dots(x, center_x):  # remove vertical dots
                        if inside_hexagon(x, y, self.image_size) and not inside_square(x, y, self.image_size,
                                                                                       self.square_side):
                            if count <= self.dot_number:
                                count += 1
                                if count > self.dot_number:
                                    dot_spacing += 1

        if dot_spacing % 2 != 0:
            return dot_spacing - 2
        return dot_spacing - 1

    def create_dot(self):
        cv_size = (1000, 1000)
        dot_spacing = self.find_space(cv_size)
        print(f"dot_spacing is: {dot_spacing}")

        """ add upper dots """
        dot_first = DotPlacer(300, 10, self.processed_image, self.hexagon_radius, self.image_size, self.square_side)
        dot_first.set_main_dot(True)
        # self._dots.append(dot_first)

        # Loop through rows and columns to create a grid of circles
        for y in range(0, cv_size[1], dot_spacing):
            for x in range(0, cv_size[0], dot_spacing):
                dot = DotPlacer(x, y, self.processed_image, self.hexagon_radius, self.image_size, self.square_side)
                if dot.inside_hexagon and not dot.inside_QRCode:
                    if self.count <= self.dot_number:
                        self._dots.append(dot)

    def show_dots(self):
        global size_arr_index

        for i, dot in enumerate(self._dots):
            if i == 0 or i == len(self._dots) - 1: dot.set_main_dot(True)
            if dot.vertical_dot and dot.y <= self.center:  # append upper vertical dots in list
                self.vertical_dots.append(dot)

            dot.show()
            self.count += 1

            # mirrored_x, mirrored_y = mirror_coordinates(dot.x, dot.y, self.center, self.center)
            # mirror_dot = DotPlacer(mirrored_x, mirrored_y, self.processed_image, self.hexagon_radius, self.image_size, self.square_side)
            # mirror_dot.show()

        """ add lower dots """
        # dot_end = DotPlacer(300, 650, self.processed_image, self.hexagon_radius, self.image_size, self.square_side)
        # dot_end.set_main_dot(True)
        # dot_end.show()

    def process_and_visualize(self):
        """Generate the hexagon with dots and visualize the result."""
        self.create_image_canvas()
        # self.draw_hexagon()
        self.add_qr_code()
        self.create_dot()
        self.show_dots()

        print(f"size is: {self.size}")
        print(f"count is: {self.count}")
        print(f"size_arr is: {size_arr}")

        # Display the final image
        plt.imshow(self.processed_image, cmap="gray")
        plt.axis()
        plt.show()


# Main entry point for generating the pattern
if __name__ == "__main__":
    processor = SQEDotPatternCode(data="hello how are you? sdsdsdsd sdsdsds")
    processor.process_and_visualize()
