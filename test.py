from PIL import Image, ImageDraw

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

# Example usage
image_path = "floor_plan1.png"  # Replace with the path to your image file
coordinates = [((100, 100), (100, 200), (200, 200), (200, 100))]  # Replace with your detected coordinates
marked_image_path = add_cctv_coordinates(image_path, coordinates)
print("Image with CCTV coordinates saved at:", marked_image_path)

