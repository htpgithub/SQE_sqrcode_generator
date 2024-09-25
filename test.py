import matplotlib.pyplot as plt
import numpy as np

# Function to create hexagon vertices
def create_hexagon(center=(0, 0), radius=1):
    angles = np.linspace(0, 2 * np.pi, 7)  # 6 sides, 7 points (start and end at the same point)
    x_hexagon = center[0] + radius * np.cos(angles)
    y_hexagon = center[1] + radius * np.sin(angles)
    return x_hexagon, y_hexagon

# Function to draw an empty hexagon
def draw_empty_hexagon(center=(0, 0), radius=1):
    # Get hexagon vertices
    x_hexagon, y_hexagon = create_hexagon(center, radius)

    # Plot hexagon outline
    plt.figure(figsize=(6, 6))
    plt.plot(x_hexagon, y_hexagon, 'black', linewidth=3)  # Draw the hexagon outline
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio equal
    plt.xlim([-radius - 1, radius + 1])
    plt.ylim([-radius - 1, radius + 1])
    plt.axis('off')  # Turn off the axis
    plt.show()

# Draw a hexagon with a given center and radius
draw_empty_hexagon(center=(0, 0), radius=5)
