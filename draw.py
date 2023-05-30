from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import cv2
from matplotlib import pyplot as plt

def input_scale():
    scale = input("enter scale: ")
    return scale

def mark_node():
    plt.imshow(Image.open(image_path))

    coordinates_list = []

    for x in range(4):
        plt.show()
        xc = input("Enter xc: ")
        if int(xc) == -1:
            break
        yc = input("Enter yc: ")
        coordinates_list.append((int(xc),int(yc)))
        print(coordinates_list)

    # 237,186 237,502
    # 991,186 991,502

    return coordinates_list

from PIL import Image, ImageFilter, ImageOps

def preprocess_image(image_path):
    # Open the image file
    image = Image.open(image_path)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Sharpen the image
    # Enhance image sharpness
    enhancer = ImageEnhance.Sharpness(grayscale_image)
    sharpened_image = enhancer.enhance(2)  # Increase sharpness by a factor of 2
    # sharpened_image = grayscale_image.filter(ImageFilter.SHARPEN)

    # Reduce noise using a filter (adjust parameters as needed)
    enhanced_image = sharpened_image.filter(ImageFilter.MedianFilter(size=1))

    # # Threshold the image to create a binary black and white image
    # threshold = 128  # Adjust the threshold value as needed
    # binary_image = denoised_image.point(lambda p: 0 if p < threshold else 255, mode="1")

    # # Enhance the image using histogram equalization
    # enhanced_image = ImageOps.equalize(binary_image)

    # Show the enhanced image
    # enhanced_image.show()

    # Save the image with the CCTV markers
    enhanced_image_path = "enhanced_image.png"  # Path to save the marked image
    enhanced_image.save(enhanced_image_path)

    return enhanced_image_path

def add_cctv_coordinates(image_path, coordinates):
    # Open the image file
    image = Image.open(image_path)

    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Iterate over the coordinates and add the CCTV markers
    for coord in coordinates:
        # Calculate the center point of the rectangle
        center_x = int((coord[0][0] + coord[2][0]) / 2)
        center_y = int((coord[0][1] + coord[2][1]) / 2)

        print(coord[0][1])
        print(coord[2][1])

        print(center_x)
        print(center_y)

        # Draw a small triangle as the CCTV coordinate
        triangle_size = 5
        triangle_coords = [
            (center_x, center_y - triangle_size),
            (center_x - triangle_size, center_y + triangle_size),
            (center_x + triangle_size, center_y + triangle_size)
        ]
        draw.polygon(triangle_coords, fill="red")

        # Draw a circle around the triangle with a distance of 10 pixels
        circle_radius = 50
        circle_coords = (
            center_x - circle_radius,
            center_y - circle_radius,
            center_x + circle_radius,
            center_y + circle_radius
        )
        draw.ellipse(circle_coords, outline="red")

    # Save the image with the CCTV markers
    marked_image_path = "marked_image.png"  # Path to save the marked image
    draw_image.save(marked_image_path)

    return marked_image_path

# ---------- ---------- ---------- ---------- ----------

# Example usage
image_path = "art.png"  # Replace with the path to your image file
coordinates = []
# coordinates.append(mark_node())
# print(coordinates)

coordinates = [((100, 100), (100, 200), (200, 200), (200, 100))]  # Replace with your detected coordinates

enhanced_image_path = preprocess_image(image_path)

marked_image_path = add_cctv_coordinates(image_path, coordinates)
print("marked image saved at:", marked_image_path)