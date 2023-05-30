from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

# os.remove(file)

# declaration
coords = []
value = []
default_scale = 1

CCTV_RADIUS = 1000
MIN_DISTANCE = 900
MAX_DISTANCE = 1100

# read floor plan image -------------------------
def read_image():
	img = cv2.imread('art.png', 1)
	return img

# preprocess image ------------------------- ???
def binarization():
	img = read_image()
	w, h, _ = img.shape
	for i in range(w):
		for j in range(h):
			if img[i,j][0] > 128 | img[i,j][1] > 128 | img[i,j][2] > 128:
				img[i,j][0] = 0
				img[i,j][1] = 0
				img[i,j][2] = 0
			else:
				img[i,j][0] = 255
				img[i,j][1] = 255
				img[i,j][2] = 255
	
	cv2.imwrite("bw.png", img)

# obtain coordinate for polygon -------------------------
def draw_rectangle(image,coords):
	# get start_point from odd & end_point from even array
	
	# img = cv2.imread('art.png', 1)
	img = image

	# print(coords[0]) # DONE
	start_point = coords[0]

	# print(coords[1]) # DONE
	end_point = coords[1]

	color = (0,0,255)
	thickness = 1

	img = cv2.rectangle(img, start_point, end_point, color, thickness)
		
	return img

# function to display the coordinates of
# of the points clicked on the image

def click_event(event, x, y, flags, params):

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)
		coords.append((int(x),int(y)))

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font,
					0.5, (0, 0, 255), 1)
		cv2.imshow('image', img)

	# checking for right mouse clicks	
	if event==cv2.EVENT_RBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)
		coords.append((int(x),int(y)))

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 0.5,
					(0, 0, 255), 1)
		cv2.imshow('image', img)

# define available area from polygon created -------------------------

# 
def area_remover(img):
	# setting mouse handler for the image
	# displaying the image
	cv2.imshow('image', img)

	# setting mouse handler for the image
	# and calling the click_event() function
	cv2.setMouseCallback('image', click_event)

	# wait for a key to be pressed to exit
	if cv2.waitKey(0) & 0xFF == ord('p'):
		if len(coords) > 1: # exit if no or 1 coord
			while len(coords) > 1:
				img = draw_rectangle(img,coords)
				del coords[0]
				del coords[0]
				if len(coords) == 0:
					break
	
	return img

# area value assigner APPROVED
def area_valuer(img):
	w, h, _ = img.shape

	for i in range(w):
		col = []
		for j in range(h):
			if 	img[i,j][2] < 128:
				col.append(1) # WALL
			else:
				col.append(0) # CLEAR
		value.append(col)
	# print(value)
				
	pd.DataFrame(value).to_csv('sample.csv')

	return value

def selectROI_area(image):
	r = cv2.selectROI("select the area", image)

	image = cv2.rectangle(image, (int(r[0]),int(r[1])), 
		(int(r[0]+r[2]),int(r[1]+r[3])), (0,0,64), -1)

	pd.DataFrame(value).to_csv('ROI.csv',index=False,header=False)
	cv2.imwrite('ROI.png', image)

	return image

# initialize cctv number based on area? -------------------------
def cctv_quantity_initializer(img,scale):
	wall = 0
	quantity = 0
	
	w, h, _ = img.shape

	for i in range(w):
		for j in range(h):
			if 	img[i,j][2] < 128: # red
				# WALL
				wall += 1

	clear_area = (h * w) - wall

	quantity = round(clear_area / (math.pi * math.pow(CCTV_RADIUS/scale,2)))
	
	return quantity

# ------------------------------------------------------------------------------------------------------

# driver function
if __name__=="__main__":
	# scale 1pixel:10cm  or 20pixel:6feet

	# reading the image
	img = read_image()
	image_path = "art.png"

	# identify image size
	h, w, _ = img.shape
	print('width: ', w)
	print('height:', h)

	stop = False
	# img = area_remover(img)
	img = selectROI_area(img)
	while stop == False:
		print("Press 'n' to stop selecting")

		if cv2.waitKey(0) & 0xFF == ord('n'):
			stop = True
		else:
			img = selectROI_area(img)

	# img = selectROI_area(img)

	# assign value to coord
	# 1 = wall, remove
	# 0 = empty
	# validation only
	value = area_valuer(img)

	print(value[:2])

	cv2.imwrite('after.png', img)

	cctv_quantity = print(cctv_quantity_initializer(img,10))