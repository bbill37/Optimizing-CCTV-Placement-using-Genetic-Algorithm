from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import cv2
import math
# from matplotlib import pyplot as plt
# import numpy
import pandas as pd
# import os
import random
# import numpy as np
from array import *

# os.remove(file)

# declaration
coords = []
value = []
rand_list = []
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
	k = cv2.waitKey(0)
	if k == ord("p"):
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

	rows, cols = (h, w)
	r,c = 0,0

	# arr = [[0 for x in range(10)] for y in range(5)]
	# arr[0][4] = 1
	# arr[4][0] = 1
	# print(arr)

	arr = [[0 for x in range(w)] for y in range(h)]
	# arr = [[0 for y in range(h)] for x in range(w)]
	# arr = []
	row = []
	col = []

	# for x in range(w):
	# 	for y in range(h):
	# 		if 	img[x,y][2] < 128:
	# 			col.append(1)
	# 		else:
	# 			col.append(0)
	# 	arr.append(col)

	for y in range(h):
		for x in range(w):
			if 	img[x,y][2] < 128:
				arr[y][x] = 1
				# row.append(1)
			else:
				arr[y][x] = 0
				# row.append(0)
		# arr.append(row)

		
	value = [[0 for x in range(w)] for y in range(h)]
	# value = arr[:][:]

	# print((value[-1][-1]))

				
	if pd.DataFrame(arr).to_csv('checker.csv') == True:
		print("checker.csv saved")

	# print(value[0][:10])
	# print(value)

	# Image.fromarray(arr).save('array.png')

	return arr

def selectROI_area(image):
	r = cv2.selectROI("select the area", image)

	# method 1
	# for y in range(r[3]-r[1]):
	# 	for x in range(r[2]-r[0]):
	# 		if image[x,y][2] > 128:
	# 			image[x,y][2] = 64

	# method 2
	image = cv2.rectangle(image, (int(r[0]),int(r[1])), 
		(int(r[0]+r[2]),int(r[1]+r[3])), (0,0,64), -1)

	# pd.DataFrame(value).to_csv('ROI.csv',index=False,header=False)
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
	
	print("\nInitializing cctv quantity: ", quantity)
	return quantity

# genetic algorithm ----------------------------------------

def rand_coords(max_cctv,img,randvalue):
	w, h, _ = img.shape

	# randvalue = [[0 for x in range(w)] for y in range(h)]

	# for y in range(h):
	# 	for x in range(w):
	# 		if 	img[x,y][2] < 128:
	# 			randvalue[y][x] = 1

	# for x in range(w):
	# 		if 	img[x,0][2] < 128:
	# 			randvalue[0][x] = 1

	randx = random.randint(0,w)
	randy = random.randint(0,h)
	rand = (randx,randy)

	# row = []
	# col = [1,1,1,0,0]
	# row.append(col)
	# col = [2,2,2,2,0]
	# row.append(col)
	# col = [3,3,3,0,0]
	# row.append(col)
	
	
	# print(row)
	# print(row[2][3])

	# randvalue[99][99] = 10

	# print(randvalue[99][99])

	print(len(randvalue))
	print(len(randvalue[0]))
	print(randy)
	print(randx)

	# print(randvalue[18][15])
	# print(randvalue[18][16])
	# print(randvalue[17][16])

	

	# n = range(len(randvalue[0]))
	# for i in n:
	# 	print(i)

	# exit()

	while len(rand_list) < max_cctv: # cctv quan

		if rand in rand_list:
			randx = random.randint(0,w)
			randy = random.randint(0,h)
			rand = (randx,randy)
		else:
			print(randy)
			print(randx)
			# if randvalue[int(randy)][int(randx)] == 1:
			if randvalue[randy][randx] == 1: # ERROR index out of range
			# if randvalue[randx][randy] == 1: # ERROR index out of range
				randx = random.randint(0,w)
				randy = random.randint(0,h)
				rand = (randx,randy)
			else:
				# if len(rand_list) > 1:
				# 	if math.dist(rand_list[-2],rand_list[-1]) < MIN_DISTANCE:
				# 		randx = random.randint(0,w)
				# 		randy = random.randint(0,h)
				# 		rand = (randx,randy)
				# 	else:
				# 		rand_list.append(rand)
				# else:
				### CONDITIONAL DISTANCE CANT WORK HERE, 
				# SO PUT IN FITNESS AS MUTATION MAYBE?
					rand_list.append(rand)
				
	print("\n")
	print(rand_list)
	return rand_list

# ------------------------------------------------------------------------------------------------------

# driver function
if __name__=="__main__":
	# scale 1pixel:10cm  or 20pixel:6feet

	# reading the image
	original_image = read_image()
	image_path = "art.png"

	# identify image size
	h, w, _ = original_image.shape
	# print('width: ', w)
	# print('height:', h)

	# img = area_remover(img)
	availableArea_image = selectROI_area(original_image)
	
	stop = False
	while stop == False:
		print("\nPress 's' to stop")

		k = cv2.waitKey(0)
		if k == ord("s"):
			stop = True
			print("\nCalculating ...")
		else:
			availableArea_image = selectROI_area(availableArea_image)

	# img = selectROI_area(img)

	# assign value to coord
	# 1 = wall, remove
	# 0 = empty
	# validation only
	value = area_valuer(availableArea_image)

	# print(len(value))
	# print(len(value[0]))

	# print(value[:2])

	if cv2.imwrite('after.png', availableArea_image) == True:
		print("\nafter.png saved")

	cctv_quantity = cctv_quantity_initializer(availableArea_image,scale = 10)

	rand_coords(cctv_quantity,availableArea_image,value)

	# test = [(1,1),(5,5),(10,10)]

	# print (math.dist(test[0],test[2]))

# ---------------------------------------------------------
