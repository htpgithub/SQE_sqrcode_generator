import numpy as np
from PIL import Image, ImageDraw
import random
import qrcode


def generate_qr_code(data):
    """    
    Generate a QR code image from input data.

    This function creates a QR code using the input data and returns an Image object 
    representing the QR code. The QR code's version, error correction level, box size, 
    and border are set according to specified parameters.

    Parameters:
    -----------
    data : str
        The data to be encoded into the QR code. Typically a URL, text, or other data.

    Returns:
    --------
    Image
        An Image object representing the generated QR code.
"""
    
    
    qr = qrcode.QRCode(
        version=10,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,  # this can be adjust box size to fit the desired dimensions
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill="black", back_color="white")
    
    # qr_img = qr_img.convert("RGB")  #ensure QR code is in RGB mode
    # qr_img = qr_img.resize((size, size), Image.LANCZOS)

    return qr_img


def create_hexagon_with_random_points(qr_img, file_name, num_points=1000):
    """
    Create a hexagonal image with a central QR code and randomly distributed bold points.

    This function generates an image with a hexagonal boundary, a QR code in the center,
    and a specified number of bold circles placed randomly outside the QR code but
    within the hexagonal boundary
    
    Parameters:
    -----------
    qr_img : PIL.Image.Image
        The QR code image to be placed in the center of the hexagon. It should be a PIL 
        Image object.

    file_name : str
        The name of the file to save the generated image, including the extension (e.g., 
        'hexagonal_random_points.png').

    num_points : int, optional, default=1000
        The number of random points to be placed outside the QR code within the hexagon.

    Returns:
    --------
    None
        The function does not return any value. It saves the generated image to the 
        specified file and writes the coordinates of the random points to 
        'circle_coordinates.txt'.
    """

    hex_size = 800  # Size of the hexagon shape
    square_size = 400  # Size of the inner square which is the qrcode

    circle_radius = 3  # radius of small circles
    circle_coordinates = []
    coord_list = []  # list of coordinates if you want to print it out

    bold_thickness = 3  # thickness of the bold circles

    qr_img = qr_img.resize((square_size, square_size), Image.LANCZOS)
    img = Image.new("RGB", (hex_size, hex_size), "white")
    draw = ImageDraw.Draw(img)

    # calculating the vertices of the hexagon, aligned vertically
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

    # draw the inner square for the QR code
    square_top_left = (center_x - square_size // 2, center_y - square_size // 2)
    square_bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    draw.rectangle([square_top_left, square_bottom_right], outline="black")

    # placing the QR code inside the inner square
    qr_position = (
        square_top_left[0],
        square_top_left[1],
        square_bottom_right[0],
        square_bottom_right[1],
    )
    img.paste(qr_img, qr_position)

    # define the square boundaries for the QR code
    square_boundaries = [
        square_top_left,
        (square_top_left[0], square_bottom_right[1]),
        square_bottom_right,
        (square_bottom_right[0], square_top_left[1]),
    ]

    # generate random points within the hexagon and draw bold circles outside the square
    while len(circle_coordinates) < num_points:
        x = random.uniform(center_x - radius, center_x + radius)
        y = random.uniform(center_y - radius, center_y + radius)

        # called on the function to check for points available in the hexagon and not in the square
        if point_inside_polygon((x, y), vertices) and not point_inside_square(
            (x, y), square_boundaries
        ):
            for t in range(bold_thickness):
                draw.ellipse(
                    [
                        x - circle_radius - t,
                        y - circle_radius - t,
                        x + circle_radius + t,
                        y + circle_radius + t,
                    ],
                    outline="black",
                    fill="black",
                )
            relative_x = x - center_x
            relative_y = y - center_y
            circle_coordinates.append((relative_x, relative_y))

    # save circle coordinates to a text file just in case it is needed
    with open("circle_coordinates.txt", "w") as file:
        for coord in circle_coordinates:
            coord_list.append(coord)
            file.write(f"{coord[0]}, {coord[1]}\n")

    img.save(file_name)
    print(f"Bitmap image with random points saved as {file_name}")
    print("Circle coordinates saved to 'circle_coordinates.txt'")


def point_inside_polygon(point, vertices):
    """
    Determine if a point is inside a given polygon using the ray-casting algorithm.

    This function checks whether a specified point lies within a polygon, 
    which is represented by a list of vertices. The algorithm used is 
    the ray-casting algorithm, which counts how many times a horizontal 
    ray originating from the point intersects the polygon's edges.

    Parameters:
    -----------
    point : tuple[float, float]
        A tuple representing the x and y coordinates of the point to be checked.

    vertices : list[tuple[float, float]]
        A list of tuples where each tuple represents the x and y coordinates 
        of a vertex of the polygon.

    Returns:
    --------
    bool
        Returns True if the point lies inside the polygon; otherwise, False.
    """
    
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
    """
    Determine if a point is inside a specified square.

    This function checks whether a given point lies within the boundaries 
    of a square. The square is defined by a list of tuples, where each tuple 
    represents the coordinates of the square's corners.

    Parameters:
    -----------
    point : tuple[float, float]
        A tuple representing the x and y coordinates of the point to be checked.

    boundaries : list[tuple[float, float]]
        A list of four tuples, each representing the x and y coordinates of 
        the square's corners in a clockwise or counterclockwise order.

    Returns:
    --------
    bool
        Returns True if the point lies inside the square; otherwise, False.
    """
    x, y = point
    (x1, y1) = boundaries[0]
    (x2, y2) = boundaries[2]

    return x1 <= x <= x2 and y1 <= y <= y2


if __name__ == "__main__":
    
    """
    Main execution block to generate a QR code and create a hexagonal image with random points.

    This block demonstrates the use of the 'generate_qr_code' and 'create_hexagon_with_random_points' 
    functions. It generates a QR code using a specified URL and then creates an image with a hexagonal 
    shape and random points outside the QR code.

    Example:
    --------
    To execute the main block, run the script as follows:
    
    $ python main.py

    This will generate a file named 'hexagonal-sqrcode-bitmap.png' with a hexagon containing a QR 
    code and random points outside the QR code.
    """
    
    # generate the QR code, the companies link is just use for testing purpose
    qr_code = generate_qr_code("https://sqe.io")
    # create the hexagonal image with random points and QR code
    create_hexagon_with_random_points(qr_code, "hexagonal-sqrcode-bitmap.png")
