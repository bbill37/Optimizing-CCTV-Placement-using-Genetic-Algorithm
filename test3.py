from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import remove_small_objects

# Read the image
image_path = "parking.jpg"
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

# Display the original image and the filtered polygons
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(image)
axs[0].set_title("Original Image")
for polygon in polygons:
    axs[0].plot(*zip(*polygon), color='red', linewidth=2)
axs[1].imshow(binary_image, cmap='gray')
axs[1].set_title("Binary Image")
plt.show()

# Save the result with filtered polygons
result_image = Image.fromarray(binary_image)

# Remove noise (isolated pixels) from the result image
result_image_array = np.array(result_image)
filtered_image_array = remove_small_objects(result_image_array, min_size=3, connectivity=1)
filtered_result_image = Image.fromarray(filtered_image_array)

# Save the filtered result image
filtered_result_image_path = "filtered_result.png"
filtered_result_image.save(filtered_result_image_path)
print("Filtered result image saved at:", filtered_result_image_path)
