import cv2
import qrcode
import numpy as np
import matplotlib.pyplot as plt
import json


class SQEDotPatternCode:
    def __init__(self, data, hexagon_radius=500, dot_spacing=25, padding=10):
        self.data = data
        self.hexagon_radius = hexagon_radius
        self.dot_spacing = dot_spacing
        self.dot_diameter = (
            0.8 * dot_spacing
        )  # Dot diameter is now 80% of the dot spacing
        self.padding = (
            padding  # Padding to prevent dots from touching the square/QR code
        )
        self.image_size = 2 * hexagon_radius + 10
        self.processed_image = None
        self.bits, self.binary_json = self.text_to_bits_and_json(data)
        self.black_dot_coords = []

    def char_to_custom_bits(self, char):
        """Convert character to a custom binary string with MSB set to '1' and extend with 0101... pattern if necessary."""
        binary_string = format(ord(char), "08b")
        binary_string = "1" + binary_string[1:]  # Ensure the first bit is always '1'

        # Extend with 0101... pattern if the string is longer than 8 bits
        if len(binary_string) > 8:
            additional_bits = ""
            while len(binary_string) % 8 != 0:
                additional_bits += "01"
                binary_string += additional_bits[: 8 - len(binary_string) % 8]

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
            np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
        )

    def draw_hexagon(self):
        """Draw a hexagon shape and its border on the image canvas with a tip pointing upwards."""
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

        # Draw hexagon border
        # cv2.polylines(self.processed_image, [vertices], isClosed=True, color=0, thickness=2)

    def draw_square(self):
        """Draw a square at the center of the hexagon and add a border."""
        # Set square side to be between 0.8 and 0.9 of the hexagon radius
        self.square_side = int(self.hexagon_radius * np.random.uniform(0.8, 0.9))
        start_x = (self.image_size - self.square_side) // 2
        start_y = (self.image_size - self.square_side) // 2
        end_x = start_x + self.square_side
        end_y = start_y + self.square_side

        # Draw square border with added padding
        # cv2.rectangle(self.processed_image, (start_x - self.padding, start_y - self.padding),
        #               (end_x + self.padding, end_y + self.padding), color=0, thickness=2)

    def place_dots(self):
        """Place dots to complete the hexagon shape by ensuring they don't touch the square or QR code borders."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        dot_radius = int(self.dot_diameter // 2)
        square_padding = self.padding + dot_radius + 5

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

        # Create a blank image to draw on
        self.processed_image = (
            np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
        )  # White background

        triangle_height = (
            center_x - self.hexagon_radius + square_padding - (self.square_side // 4)
        )
        for x in range(
            triangle_height, center_x - self.square_side // 2, int(self.dot_spacing)
        ):
            row_y_positions = calculate_odd_row_positions(
                center_y - (x - triangle_height), 
                center_y + (x - triangle_height),
                int(self.dot_spacing),
            )
            for y in row_y_positions:
                if bit_index < len(self.bits):
                    if self.is_inside_hexagon(x, y) and not self.is_inside_square(x, y):
                        color = 0 if self.bits[bit_index] == "1" else 255
                        if color == 0:
                            cv2.circle(
                                self.processed_image, (y, x), dot_radius, color, -1
                            )
                            self.black_dot_coords.append((y, x))
                        bit_index += 1
                        
        rectangle_start_x = (
            center_x - self.square_side // 2
        )
        rectangle_start_y = center_y + (triangle_height)

        for x in range(
            rectangle_start_x,
            rectangle_start_x + self.square_side,
            int(self.dot_spacing),
        ):
            row_y_positions = calculate_odd_row_positions(
                rectangle_start_y,
                center_y + self.hexagon_radius - square_padding,
                int(self.dot_spacing),
            )
            for y in row_y_positions:
                if bit_index < len(self.bits):
                    if self.is_inside_hexagon(x, y) and not self.is_inside_square(x, y):
                        color = 0 if self.bits[bit_index] == "1" else 255
                        if color == 0:
                            cv2.circle(
                                self.processed_image, (y, x), dot_radius, color, -1
                            )
                            self.black_dot_coords.append((y, x))
                        bit_index += 1

        # Function to rotate an image 180 degrees around its vertical axis (flip horizontally)
        def flip_horizontally(image):
            return cv2.flip(image, 1)

        # Function to flip, rotate, and merge the image by overlaying
        def flip_rotate_and_overlay_image(image):
            # Flip the image upside down
            flipped_image = cv2.flip(image, 0)

            # Rotate the flipped image by 180 degrees around the vertical axis (flip horizontally)
            rotated_flipped_image = flip_horizontally(flipped_image)

            # Overlay the rotated and flipped image on the original image
            combined_image = np.minimum(image, rotated_flipped_image)
            return combined_image

        # Flip, rotate, and overlay to create the complete hexagon
        self.processed_image = flip_rotate_and_overlay_image(self.processed_image)

        # Generate the QR code
        qr_text = self.data
        qr = qrcode.QRCode(border=0)
        qr.add_data(qr_text)
        qr.make(fit=True)

        # Convert QR code to an image
        qr_image = qr.make_image(fill="black", back_color="white").convert("L")
        qr_image = np.array(qr_image)

        # Resize QR code to fit within the central square of the hexagon
        qr_image_resized = cv2.resize(
            qr_image, (self.square_side, self.square_side), interpolation=cv2.INTER_AREA
        )

        # Overlay QR code at the center of the hexagon
        qr_top_left_x = center_x - self.square_side // 2
        qr_top_left_y = center_y - self.square_side // 2

        for i in range(qr_image_resized.shape[0]):
            for j in range(qr_image_resized.shape[1]):
                if qr_image_resized[i, j] == 0:
                    x = qr_top_left_x + i
                    y = qr_top_left_y + j
                    self.processed_image[x, y] = 0

        # Draw the border around the hexagon
        hexagon_points = [
            (
                center_x + self.hexagon_radius * np.cos(np.pi / 6 + np.pi / 3 * i),
                center_y + self.hexagon_radius * np.sin(np.pi / 6 + np.pi / 3 * i),
            )
            for i in range(6)
        ]
        hexagon_points = np.array(hexagon_points, np.int32)
        cv2.polylines(
            self.processed_image, [hexagon_points], isClosed=True, color=0, thickness=2
        )

        # # Draw the border around the QR code
        # cv2.rectangle(self.processed_image, (qr_top_left_y - self.padding, qr_top_left_x - self.padding),
        #   (qr_top_left_y + self.square_side + self.padding, qr_top_left_x + self.square_side + self.padding), color=0, thickness=2)

    def is_inside_square(self, x, y):
        """Check if a point (x, y) is inside the central square."""
        start_x = (self.image_size - self.square_side) // 2 - self.padding
        start_y = (self.image_size - self.square_side) // 2 - self.padding
        end_x = start_x + self.square_side + 2 * self.padding
        end_y = start_y + self.square_side + 2 * self.padding

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
    processor = SQEDotPatternCode(data="Quantum Secure Blockchain Platform", padding=10)
    processor.process_and_visualize()
