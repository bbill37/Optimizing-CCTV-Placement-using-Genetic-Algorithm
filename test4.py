from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read the image
image_path = "floor2.png"
image = Image.open(image_path)

# Enhance image sharpness
enhancer = ImageEnhance.Sharpness(image)
sharpened_image = enhancer.enhance(2.0)  # Increase sharpness by a factor of 2

# Convert the image to grayscale
grayscale_image = sharpened_image.convert("L")

# Convert the grayscale image to a numpy array
image_array = np.array(grayscale_image)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area and solidity
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Filter out small contours based on area threshold
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        if solidity > 0.8:  # Filter out contours with low solidity (likely hollow polygons)
            filtered_contours.append(contour)

# Convert the filtered contours to polygons
polygons = []
for contour in filtered_contours:
    contour = contour.squeeze()
    polygon = [(x, y) for x, y in contour]
    polygons.append(polygon)

# Remove noise with a maximum size of 2 pixels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Find contours in the filtered image
contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Update the filtered polygons with noise removed
filtered_polygons = []
for contour in contours:
    contour = contour.squeeze()
    polygon = [(x, y) for x, y in contour]
    filtered_polygons.append(polygon)

# Display the original image with the filtered polygons
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image)
axs[0].set_title("Original Image")
for polygon in polygons:
    axs[0].plot(*zip(*polygon), color='red', linewidth=2)

# Display the filtered image with noise removed
axs[1].imshow(filtered_image, cmap='gray')
axs[1].set_title("Filtered Image")

# Plot the filtered polygons
for polygon in filtered_polygons:
    axs[1].plot(*zip(*polygon), color='red', linewidth=2)

plt.show()
