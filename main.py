import numpy as np
from PIL import Image, ImageDraw
import random


coord_list = []


def create_hexagon_with_random_points(file_name, num_points=5000):
    hex_size = 800  # Size of the hexagon shape
    square_size = 400  # Size of the inner square
    img = Image.new("RGB", (hex_size, hex_size), "white")
    draw = ImageDraw.Draw(img)

    # Calculate vertices of the hexagon, aligned vertically
    center_x, center_y = hex_size // 2, hex_size // 2
    radius = hex_size // 2 - 20
    vertices = [
        (center_x + radius * np.cos(theta), center_y + radius * np.sin(theta))
        for theta in np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 7)
    ]

    # draw the hexagon with bold border
    for i in range(5, 0, -1):
        adjusted_vertices = [
            (
                center_x + (radius + i) * np.cos(theta),
                center_y + (radius + i) * np.sin(theta),
            )
            for theta in np.linspace(np.pi / 6, 2 * np.pi + np.pi / 6, 7)
        ]
        draw.polygon(adjusted_vertices, outline="black")

    # draw the inner square to place the generated qr code
    square_top_left = (center_x - square_size // 2, center_y - square_size // 2)
    square_bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    draw.rectangle([square_top_left, square_bottom_right], outline="black")

    # define the square boundaries for the qr code
    square_boundaries = [
        square_top_left,
        (square_top_left[0], square_bottom_right[1]),
        square_bottom_right,
        (square_bottom_right[0], square_top_left[1]),
    ]

    # Generate random points within the hexagon and draw circles outside the square
    circle_radius = 3  # Radius of small circles
    circle_coordinates = []

    while len(circle_coordinates) < num_points:
        x = random.uniform(center_x - radius, center_x + radius)
        y = random.uniform(center_y - radius, center_y + radius)
        if point_inside_polygon((x, y), vertices) and not point_inside_square(
            (x, y), square_boundaries
        ):
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            draw.ellipse(
                [
                    x - circle_radius,
                    y - circle_radius,
                    x + circle_radius,
                    y + circle_radius,
                ],
                outline=color,
            )
            relative_x = x - center_x
            relative_y = y - center_y
            circle_coordinates.append((relative_x, relative_y))

    # Save circle coordinates to a text file
    with open("circle_coordinates.txt", "w") as file:
        for coord in circle_coordinates:
            coord_list.append(coord)
            file.write(f"{coord[0]}, {coord[1]}\n")

    img.save(file_name)

    print(f"Hexagonal image with random points saved as {file_name}")
    print("Circle coordinates saved to 'circle_coordinates.txt'")


def point_inside_polygon(point, vertices):
    x, y = point
    n = len(vertices)
    inside = False

    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def point_inside_square(point, boundaries):
    x, y = point
    (x1, y1) = boundaries[0]
    (x2, y2) = boundaries[2]

    return x1 <= x <= x2 and y1 <= y <= y2


# Create the hexagonal image with random points
create_hexagon_with_random_points("hexagonal_random_points_outside_square.png")