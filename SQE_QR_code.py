import cv2
import qrcode
import numpy as np
import matplotlib.pyplot as plt
import json


class DotPlacer:
    def __init__(self, hexagon_radius, dot_spacing, dot_diameter, bits):
        self.hexagon_radius = hexagon_radius
        self.dot_spacing = dot_spacing
        self.dot_diameter = dot_diameter
        self.bits = bits
        self.black_dot_coords = []

    def mirror_coordinates(self, x, y, center_x, center_y):
        """Mirror coordinates across both x and y axes relative to the center point."""
        mirrored_x = 2 * center_x - x
        mirrored_y = 2 * center_y - y
        return mirrored_x, mirrored_y

    def is_inside_hexagon(self, x, y, image_size):
        """Check if a point (x, y) is inside the hexagon."""
        center_x = image_size // 2
        center_y = image_size // 2
        dx = abs(x - center_x)
        dy = abs(y - center_y)

        if dx > self.hexagon_radius or dy > np.sqrt(3) * self.hexagon_radius / 2:
            return False

        return dy < np.sqrt(1.5) * (self.hexagon_radius - dx)

    def is_inside_square(self, x, y, image_size, square_side):
        """Check if a point (x, y) is inside the central square."""
        start_x = (image_size - square_side) // 2
        start_y = (image_size - square_side) // 2
        end_x = start_x + square_side
        end_y = start_y + square_side

        return start_x < x < end_x and start_y < y < end_y

    def place_dots(self, processed_image, image_size, square_side):
        """Place dots to complete the hexagon shape by placing the dots in both the upper right and lower left regions."""
        center_x = image_size // 2
        center_y = image_size // 2
        dot_radius = int(self.dot_diameter // 2)
        square_padding = dot_radius + 5
        bit_index = 0

        # Function to calculate dot positions for a row with an odd number of dots
        def calculate_odd_row_positions(start, end, spacing):
            row_positions = []
            mid = (start + end) // 2
            row_positions.append(mid)
            for i in range(1, (end - start) // (2 * spacing) + 1):
                row_positions.append(mid - i * spacing)
                row_positions.append(mid + i * spacing)
            return sorted(row_positions)

        # Add 9 Vertical Centerline Dots (not part of the encoded bits)
        triangle_height = center_x - self.hexagon_radius + square_padding - (square_side // 4)
        for i in range(9):
            x = triangle_height + i * int(self.dot_spacing)
            y = center_y  # Centerline
            cv2.circle(processed_image, (y, x), dot_radius, 0, -1)  # Black dot
            self.black_dot_coords.append((y, x))

            # Mirror and add the corresponding dot on the lower left region
            mirrored_x, mirrored_y = self.mirror_coordinates(x, y, center_x, center_y)
            cv2.circle(processed_image, (mirrored_y, mirrored_x), dot_radius, 0, -1)
            self.black_dot_coords.append((mirrored_y, mirrored_x))

        for x in range(triangle_height, center_x - square_side // 2, int(self.dot_spacing)):
            row_y_positions = calculate_odd_row_positions(
                center_y - (x - triangle_height),
                center_y + (x - triangle_height),
                int(self.dot_spacing)
            )
            for y in row_y_positions:
                if bit_index < len(self.bits):
                    if self.is_inside_hexagon(x, y, image_size) and not self.is_inside_square(x, y, image_size,
                                                                                              square_side):
                        if self.bits[bit_index] == "1":
                            cv2.circle(processed_image, (y, x), dot_radius, 0, -1)
                            self.black_dot_coords.append((y, x))
                        bit_index += 1

                    mirrored_x, mirrored_y = self.mirror_coordinates(x, y, center_x, center_y)
                    if self.is_inside_hexagon(mirrored_x, mirrored_y, image_size) and not self.is_inside_square(
                            mirrored_x, mirrored_y, image_size, square_side):
                        if self.bits[bit_index - 1] == "1":
                            cv2.circle(processed_image, (mirrored_y, mirrored_x), dot_radius, 0, -1)
                            self.black_dot_coords.append((mirrored_y, mirrored_x))

        rectangle_start_x = center_x - square_side // 2
        rectangle_start_y = center_y

        for x in range(rectangle_start_x, rectangle_start_x + square_side, int(self.dot_spacing)):
            row_y_positions = calculate_odd_row_positions(rectangle_start_y,
                                                          center_y + self.hexagon_radius - square_padding,
                                                          int(self.dot_spacing))
            for y in row_y_positions:
                if bit_index < len(self.bits):
                    if self.is_inside_hexagon(x, y, image_size) and not self.is_inside_square(x, y, image_size,
                                                                                              square_side):
                        if self.bits[bit_index] == "1":
                            cv2.circle(processed_image, (y, x), dot_radius, 0, -1)
                            self.black_dot_coords.append((y, x))
                        bit_index += 1

                    mirrored_x, mirrored_y = self.mirror_coordinates(x, y, center_x, center_y)
                    if self.is_inside_hexagon(mirrored_x, mirrored_y, image_size) and not self.is_inside_square(
                            mirrored_x, mirrored_y, image_size, square_side):
                        if self.bits[bit_index - 1] == "1":
                            cv2.circle(processed_image, (mirrored_y, mirrored_x), dot_radius, 0, -1)
                            self.black_dot_coords.append((mirrored_y, mirrored_x))

        return self.black_dot_coords


class SQEDotPatternCode:
    def __init__(self, data, hexagon_radius=500, dot_spacing=35):
        self.data = data
        self.hexagon_radius = hexagon_radius
        self.dot_spacing = dot_spacing
        self.dot_diameter = 0.6 * dot_spacing  # Dot diameter is 90% of the dot-to-dot dimension
        self.image_size = 2 * hexagon_radius + 10
        self.processed_image = None
        self.bits, self.binary_json = self.text_to_bits_and_json(data)
        self.square_side = int(self.hexagon_radius * 0.8)
        self.dot_placer = DotPlacer(self.hexagon_radius, self.dot_spacing, self.dot_diameter, self.bits)

    def char_to_custom_bits(self, char):
        """Convert character to a custom binary string with MSB set to '1'."""
        binary_string = format(ord(char), "08b")
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
        qr_top_left_x = center_x - qr_image_resized.shape[0] // 2
        qr_top_left_y = center_y - qr_image_resized.shape[1] // 2

        for i in range(qr_image_resized.shape[0]):
            for j in range(qr_image_resized.shape[1]):
                if qr_image_resized[i, j] == 0:  # If the QR code pixel is black
                    x = qr_top_left_x + i
                    y = qr_top_left_y + j
                    self.processed_image[x, y] = 0  # Place the QR code pixel on the image

    def process_and_visualize(self):
        """Generate the hexagon with dots and visualize the result."""
        self.create_image_canvas()
        self.draw_hexagon()
        self.add_qr_code()
        black_dot_coords = self.dot_placer.place_dots(self.processed_image, self.image_size, self.square_side)

        coords_filename = "black_dot_coords.json"
        with open(coords_filename, "w") as coords_file:
            json.dump(black_dot_coords, coords_file, indent=4)

        print(f"Black dot coordinates saved to {coords_filename}")

        # Display the final image
        plt.imshow(self.processed_image, cmap="gray")
        plt.axis("off")
        plt.show()


# Main entry point for generating the pattern
if __name__ == "__main__":
    processor = SQEDotPatternCode(
        data="https://organicminded.org/questionAnswers"
    )
    processor.process_and_visualize()