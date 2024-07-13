import cv2
import numpy as np
import os


def rotate_and_detect_circles(image, angle):
    """
    Rotate the image by a specified angle and detect circles within the rotated image.

    This function rotates the given image by the specified angle and then uses the 
    Hough Circle Transform to detect circles within the rotated image. The coordinates 
    of the detected circles are returned relative to the center of the image.

    Parameters:
    -----------
    image : numpy.ndarray
        The image to be rotated and analyzed, expected to be in grayscale.
    angle : float
        The angle by which to rotate the image, in degrees.

    Returns:
    --------
    list of tuple
        A list of tuples where each tuple contains the x and y coordinates of the circle 
        centers relative to the center of the image. If no circles are detected, an 
        empty list is returned.
    """
    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Debug: show rotated image
    cv2.imshow(f"Rotated Image at {angle} degrees", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(rotated, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                               param1=50, param2=30, minRadius=10, maxRadius=30)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Get coordinates relative to the center of the image
        coordinates = [(x - center[0], y - center[1]) for (x, y, r) in circles]
        return coordinates
    else:
        print(f"No circles detected for angle {angle}")
        return []


if __name__ == "__main__":
    """
    Main execution block to rotate an image at specified angles and detect circles.

    This block processes a given grayscale image by rotating it at multiple specified angles 
    and detecting circles in each rotated image. The coordinates of the detected circles, 
    relative to the center of the image, are saved to text files.

    Example:
    --------
    To execute the main block, run the script as follows:
    
    $ python main.py

    This will process the image "QRCODE_COPY.jpg", rotate it at angles 0, 30, 45, 60, and 90 degrees, 
    detect circles, and save the coordinates to separate text files in the 'data' directory.
    """
    
    # Load the image
    image_path = "QRCODE_COPY.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to a manageable size
    resize_factor = 0.5
    image = cv2.resize(image, (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor)))

    # Define rotation angles
    angles = [0, 30, 45, 60, 90]

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Directory to save the text files
    output_dir = "data"

    # Create the directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each angle
    for angle in angles:
        coordinates = rotate_and_detect_circles(image, angle)
        
        # Print coordinates
        print(f"Coordinates for angle {angle}:")
        for coord in coordinates:
            print(coord)
        
        # Save coordinates to a text file
        file_name = f"coordinates_{angle}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as file:
            for coord in coordinates:
                file.write(f"{coord[0]},{coord[1]}\n")
