import cv2
import qrcode
import numpy as np
import matplotlib.pyplot as plt


class SQEDotPatternCode:
    def __init__(self, data, hexagon_radius=500, dot_spacing=25, padding=10):
        self.data = data
        self.hexagon_radius = hexagon_radius
        self.dot_spacing = dot_spacing
        self.dot_diameter = 0.8 * dot_spacing
        self.padding = padding
        self.image_size = 2 * hexagon_radius + 10
        self.processed_image = None
        self.bits = self.text_to_bits_and_json(data)
        self.black_dot_coords = []
        self.white_dots_coords = []

    def char_to_custom_bits(self, char):
        """Convert character to a custom binary string with MSB set to '1'. Ensure it fits within 9 bits."""
        binary_string = format(ord(char), "08b")
        binary_string = "1" + binary_string[1:]  # Ensure the first bit is always '1'

        if len(binary_string) > 9:
            binary_string = binary_string[:5] + "0101"

        return binary_string

    def text_to_bits_and_json(self, text):
        """Converts text to a binary string with custom MSB and returns JSON with individual characters and their bits."""
        binary_string = ""
        filename = "binary_data.txt"

        with open(filename, "w") as f:
            # check if binary bit in hte txt file
            for char in text:
                binary_data = ""
                custom_bits = self.char_to_custom_bits(char)
                binary_string += custom_bits
                binary_data = f"\t\t{char} -> {custom_bits}\n"

                f.write(binary_data)

        return binary_string

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

    def draw_square(self):
        """Draw a square at the center of the hexagon and add a border."""
        self.square_side = int(self.hexagon_radius * np.random.uniform(0.8, 0.9))
        start_x = (self.image_size - self.square_side) // 2
        start_y = (self.image_size - self.square_side) // 2
        end_x = start_x + self.square_side
        end_y = start_y + self.square_side

    def place_dots(self):
        """
        Place dots on the hexagonal pattern based on the binary data.
        Dots corresponding to '1' bits are black, while those corresponding to '0' bits are white.
        The method also takes care of dot placement within the hexagon and avoiding the central square.
        """
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        dot_radius = int(self.dot_diameter // 2)
        square_padding = self.padding + dot_radius + 5

        bit_index = 0
        total_bits = len(self.bits)

        def calculate_odd_row_positions(start, end, spacing):
            row_positions = []
            mid = (start + end) // 2
            row_positions.append(mid)
            for i in range(1, (end - start) // (2 * spacing) + 1):
                row_positions.append(mid - i * spacing)
                row_positions.append(mid + i * spacing)
            return sorted(row_positions)

        self.processed_image = (
            np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        )

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
                if bit_index < total_bits:
                    if self.is_inside_hexagon(x, y) and not self.is_inside_square(x, y):
                        color = 0 if self.bits[bit_index] == "1" else 255

                        if color == 0:
                            if bit_index < 8:
                                dot_color = (255, 0, 0)  # Red for first 8 bits
                            elif bit_index >= total_bits - 8:
                                dot_color = (0, 255, 0)  # Green for last 8 bits
                            else:
                                dot_color = 0  # Black for all other dots

                            cv2.circle(
                                self.processed_image, (y, x), dot_radius, dot_color, -1
                            )
                            self.black_dot_coords.append((y, x))
                        bit_index += 1

        rectangle_start_x = center_x - self.square_side // 2
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
                if bit_index < total_bits:
                    if self.is_inside_hexagon(x, y) and not self.is_inside_square(x, y):
                        color = 0 if self.bits[bit_index] == "1" else 255

                        if color == 0:
                            if bit_index < 8:
                                dot_color = (
                                    255,
                                    0,
                                    0,
                                )  # Red for first 8 bits to indicate the starting point
                            elif bit_index >= total_bits - 8:
                                dot_color = (
                                    0,
                                    255,
                                    0,
                                )  # Green for last 8 bits to indicate the ending point
                            else:
                                dot_color = 0  # Black for all other dots

                            cv2.circle(
                                self.processed_image, (y, x), dot_radius, dot_color, -1
                            )
                            self.black_dot_coords.append((y, x))
                        else:
                            self.white_dots_coords.append((y, x))

                        bit_index += 1

        def flip_horizontally(image):
            return cv2.flip(image, 1)

        def flip_rotate_and_overlay_image(image):
            flipped_image = cv2.flip(image, 0)
            rotated_flipped_image = flip_horizontally(flipped_image)
            combined_image = np.minimum(image, rotated_flipped_image)
            return combined_image

        self.processed_image = flip_rotate_and_overlay_image(self.processed_image)

        qr_text = self.data
        qr = qrcode.QRCode(border=2)
        qr.add_data(qr_text)
        qr.make(fit=True)

        qr_image = qr.make_image(fill="black", back_color="white").convert("L")
        qr_image = np.array(qr_image)

        qr_image_resized = cv2.resize(
            qr_image, (self.square_side, self.square_side), interpolation=cv2.INTER_AREA
        )

        qr_top_left_x = center_x - self.square_side // 2
        qr_top_left_y = center_y - self.square_side // 2

        for i in range(qr_image_resized.shape[0]):
            for j in range(qr_image_resized.shape[1]):
                if qr_image_resized[i, j] == 0:
                    x = qr_top_left_x + i
                    y = qr_top_left_y + j
                    self.processed_image[x, y] = 0

        expanded_hexagon_points = [
            (
                center_x
                + (self.hexagon_radius + self.dot_diameter)
                * np.cos(np.pi / 6 + np.pi / 3 * i),
                center_y
                + (self.hexagon_radius + self.dot_diameter)
                * np.sin(np.pi / 6 + np.pi / 3 * i),
            )
            for i in range(6)
        ]

        for i in range(len(expanded_hexagon_points)):
            x_adjustment = (expanded_hexagon_points[i][0] - center_x) * 0.01
            y_adjustment = (expanded_hexagon_points[i][1] - center_y) * 0.02
            expanded_hexagon_points[i] = (
                expanded_hexagon_points[i][0] + x_adjustment,
                expanded_hexagon_points[i][1] - y_adjustment,
            )

        expanded_hexagon_points = np.array(expanded_hexagon_points, np.int32)
        cv2.polylines(
            self.processed_image,
            [expanded_hexagon_points],
            isClosed=True,
            color=0,
            thickness=2,
        )

        print(f"Total black dots placed: {len(self.black_dot_coords)}")
        print(f"Total '1' bits: {self.bits.count('1')}")

        print(f"Total white dots placed: {len(self.white_dots_coords)}")
        print(f"Total '0' bits: {self.bits.count('0')}")

        if len(self.black_dot_coords) != self.bits.count("1"):
            print(
                "Warning: The number of placed dots does not match the number of '1' bits."
            )

        if len(self.white_dots_coords) != self.bits.count("0"):
            print(
                "Warning: The number of placed dots does not match the number of '0' bits."
            )

    def is_inside_hexagon(self, x, y):
        """Check if the point (x, y) is inside the hexagon based on the calculated vertices."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        size = self.hexagon_radius

        vertices = [
            (
                center_x + size * np.cos(np.pi / 6 + np.pi / 3 * i),
                center_y + size * np.sin(np.pi / 6 + np.pi / 3 * i),
            )
            for i in range(6)
        ]

        vertices = np.array(vertices, np.int32)
        hex_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        cv2.fillPoly(hex_mask, [vertices], 1)
        return hex_mask[x, y] == 1

    def is_inside_square(self, x, y):
        """Check if the point (x, y) is inside the central square."""
        center_x = self.image_size // 2
        center_y = self.image_size // 2
        half_side = self.square_side // 2
        return (
            center_x - half_side <= x <= center_x + half_side
            and center_y - half_side <= y <= center_y + half_side
        )

    def save_image(self, filename="pattern_with_qr.png"):
        """Save the generated dot pattern image to a file."""
        if self.processed_image is not None:
            cv2.imwrite(filename, self.processed_image)
            print(f"Image saved as {filename}")

    def plot_image(self):
        """Plot the generated image using Matplotlib."""
        sqe.create_image_canvas()
        sqe.draw_hexagon()
        sqe.draw_square()
        sqe.place_dots()
        sqe.save_image()
        if self.processed_image is not None:

            plt.imshow(self.processed_image, cmap="gray")
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    sqe = SQEDotPatternCode(
        data="Welcome To SQE, The Quantum Secure Blockchain Platform"
    )
    sqe.plot_image()

    num_black_dots = sqe.bits.count("1")
    num_white_dots = sqe.bits.count("0")

    print(f"Number of 1's: {num_black_dots}")
    print(f"Number of 0's: {num_white_dots}")
