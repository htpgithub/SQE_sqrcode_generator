import cv2
import numpy as np
import matplotlib.pyplot as plt
from reedsolo import RSCodec
import json

class HexagonDotPattern:
    def __init__(self, data, hexagon_radius=500, dot_diameter=25, dot_spacing=25):
        self.data = data
        self.hexagon_radius = hexagon_radius
        self.dot_diameter = dot_diameter
        self.dot_spacing = dot_spacing
        self.image_size = 2 * hexagon_radius + 50
        self.processed_image = None
        self.bits, self.binary_json = self.text_to_bits_and_json(data)
        self.black_dot_coords = []

    def char_to_custom_bits(self, char):
        """Convert character to a custom binary string with MSB set to '1'."""
        binary_string = format(ord(char), "08b")
        binary_string = '1' + binary_string[1:]

        if len(binary_string) > 8:
            while len(binary_string) % 8 != 0:
                binary_string += '01'[(len(binary_string) % 2):((len(binary_string) % 2) + 1)]
        
        return binary_string

    def text_to_bits_and_json(self, text):
        """Converts text to a binary string with custom MSB and returns JSON with individual characters and their bits."""
        binary_string = ""
        binary_dict = {}

        for char in text:
            custom_bits = self.char_to_custom_bits(char)
            binary_string += custom_bits
            binary_dict[char] = custom_bits

        # Convert the dictionary to JSON
        json_filename = "binary_data.json"
        with open(json_filename, "w") as json_file:
            json.dump(binary_dict, json_file, indent=4)
        
        print(f"Binary data saved to {json_filename}")
        return binary_string, binary_dict

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

    def draw_square(self):
        """Draw a square at the center of the hexagon."""
        self.square_side = int(self.hexagon_radius * 0.9)
        start_x = (self.image_size - self.square_side) // 2
        start_y = (self.image_size - self.square_side) // 2
        end_x = start_x + self.square_side
        end_y = start_y + self.square_side

    def place_dots(self):
        """Place dots in Quadrants, then mirror them across the other quadrants."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        dot_radius = self.dot_diameter // 2
        square_padding = dot_radius + 5 

        bit_index = 0

        for x in range(center_x + square_padding, center_x + self.hexagon_radius - self.dot_spacing, self.dot_spacing):
            for y in range(center_y - self.hexagon_radius + square_padding, center_y, self.dot_spacing):

                if bit_index < len(self.bits):
                    if self.is_inside_hexagon(x, y) and not self.is_inside_square(x, y):
                        color = 0 if self.bits[bit_index] == "1" else 255
                        cv2.circle(
                            self.processed_image, (y, x), dot_radius, color, -1
                        )

                        if color == 0:
                            self.black_dot_coords.append((y, x))

                        mirrored_y1, mirrored_x1 = center_y - (y - center_y), x
                        cv2.circle(
                            self.processed_image,
                            (mirrored_y1, mirrored_x1),
                            dot_radius,
                            color,
                            -1,
                        )
                        if color == 0:
                            self.black_dot_coords.append((mirrored_y1, mirrored_x1))

                        mirrored_y2, mirrored_x2 = y, center_x + (center_x - x)
                        cv2.circle(
                            self.processed_image,
                            (mirrored_y2, mirrored_x2),
                            dot_radius,
                            color,
                            -1,
                        )
                        if color == 0:
                            self.black_dot_coords.append((mirrored_y2, mirrored_x2))

                        mirrored_y3, mirrored_x3 = center_y - (y - center_y), center_x + (center_x - x)
                        cv2.circle(
                            self.processed_image,
                            (mirrored_y3, mirrored_x3),
                            dot_radius,
                            color,
                            -1,
                        )
                        if color == 0:
                            self.black_dot_coords.append((mirrored_y3, mirrored_x3))

                        bit_index += 1

    def is_inside_square(self, x, y):
        """Check if a point (x, y) is inside the central square."""
        start_x = (self.image_size - self.square_side) // 2
        start_y = (self.image_size - self.square_side) // 2
        end_x = start_x + self.square_side
        end_y = start_y + self.square_side

        return start_x < x < end_x and start_y < y < end_y

    def is_inside_hexagon(self, x, y):
        """Check if a point (x, y) is inside the hexagon."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        dx = abs(x - center_x)
        dy = abs(y - center_y)

        if dx > self.hexagon_radius or dy > np.sqrt(3) * self.hexagon_radius / 2:
            return False

        return dy <= np.sqrt(3) * (self.hexagon_radius - dx)

    def process_and_visualize(self):
        """Generate the hexagon with dots and visualize the result."""
        self.create_image_canvas()
        self.draw_hexagon()
        self.draw_square()
        self.place_dots()

        # Save black dot coordinates to a JSON file
        coords_filename = "black_dot_coords.json"
        with open(coords_filename, "w") as coords_file:
            json.dump(self.black_dot_coords, coords_file, indent=4)

        print(f"Black dot coordinates saved to {coords_filename}")

        # Display the final image
        plt.imshow(self.processed_image, cmap="gray")
        plt.axis("off")
        plt.show()


# Main entry point for generating the pattern
if __name__ == "__main__":
    processor = HexagonDotPattern(
        data="Welcome To SQE, The Quantum Secure Blockchain Platform"
    )
    processor.process_and_visualize()
