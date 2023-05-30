# Python program to explain cv2.circle() method
	
# importing cv2
import cv2
	
# path
# path = r'C:\Users\Rajnish\Desktop\geeksforgeeks\geeks.png'
	
# Reading an image in default mode
image = cv2.imread("art.png")
	
# Window name in which image is displayed
window_name = 'Image'

# Center coordinates
center_coordinates = (120, 50)

# Radius of circle
radius = 20

# Blue color in BGR
color = (127, 127, 0)

# Line thickness of 2 px
thickness = 1

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
image = cv2.circle(image, center_coordinates, radius, color, thickness)

# Displaying the image
cv2.imshow(window_name, image)
cv2.waitKey(0)