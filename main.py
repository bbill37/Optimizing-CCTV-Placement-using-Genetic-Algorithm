from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import cv2
import math
# from matplotlib import pyplot as plt
import pandas as pd
import sys
import random
import numpy as np
from array import *

# ------------------------------------------------------------

# draw small dot as coordinate
def circleCoord(img, center, radius, color, thickness):
    image = cv2.circle(img, (center[1],center[0]), radius, color, thickness)
    return image

# draw filled circle as area with coordinate as centroid
def circleArea(img, center, radius, color, thickness):
    image = cv2.circle(img, (center[1],center[0]), radius, color, thickness)
    return image

# put text for coordinate
def putTextCoord(img, text, org, fontFace, fontScale, color, thickness):
    image = cv2.putText(img, str(text[1])+','+str(text[0]), (org[1],org[0]), fontFace, fontScale, color, thickness)
    return image

# calculate slope between two coordinates to increase distance between them
def slope(x1, y1, x2, y2):
    if(x2 - x1 != 0):
      return (float)(y2-y1)/(x2-x1)
    return sys.maxint

# penalty for GA if distance between coordinates less than minimum distance
def coord_penalty(img, coord):
	w,h,_=img.shape

	for x in range(0,w):
		for y in range(0,h):
			if 	coord_vals[x][y] == 1:
				if math.dist((int(x),int(y)),coord) < (MIN_DISTANCE/default_scale):
					if int(x) < coord[0] and int(y) < coord[1]:
						coord = ((coord[0]+1),(coord[1]+1))

					if int(x) < coord[0] and int(y) > coord[1]:
						coord = ((coord[0]+1),(coord[1]-1))

					if int(x) > coord[0] and int(y) < coord[1]:
						coord = ((coord[0]-1),(coord[1]+1))

					if int(x) > coord[0] and int(y) > coord[1]:
						coord = ((coord[0]-1),(coord[1]-1))

					# coord_vals[x][y] = 1
					# img[x,y] = (7, 0, 11)

# ------------------------------------------------------------

# declaration
raw_path = "art.png"
coords = []
value = []
rand_list = []
default_scale = 1

img = cv2.imread(raw_path, 1)
w, h, c = img.shape
coord_vals = [[0 for y in range(0,h)] for x in range(0,w)]

CCTV_RADIUS = 1000
MIN_DISTANCE = 900
MAX_DISTANCE = 1100

# read floor plan image -------------------------
def read_image():
	img = cv2.imread(raw_path, 1)
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
	
	# img = cv2.imread(raw_path, 1)
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
	img = cv2.imread(raw_path, 1)

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

	# cv2.imwrite('availableArea.png',img)
	
	return img

# area value assigner APPROVED
def area_valuer(img):
	w, h, _ = img.shape

	for x in range(0,w):
		for y in range(0,h):
			if 	(img[x,y][0] < 64 and img[x,y][1] < 64 and img[x,y][2] < 128):
				# walls
				coord_vals[x][y] = 1
				img[x,y] = (7, 0, 11)
			else:
				coord_vals[x][y] = 0
				img[x,y] = (255,255,255)

			if 	(img[x,y][0] < 128 and img[x,y][1] > 128 and img[x,y][2] < 128):
				# coverage
				coord_vals[x][y] = 2

	cv2.imwrite('value.png',img)

	if pd.DataFrame(coord_vals).to_csv('coord_vals.csv')==True: 
		print("coord_vals.csv saved")

	return img

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
	cv2.imwrite('availableArea.png',image)

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

	quantity = round(clear_area / (math.pi * math.pow((CCTV_RADIUS/scale),2)))
	
	print("\nInitializing cctv quantity: ", quantity)
	return quantity

def rand_coords(max_cctv,img):
	w, h, _ = img.shape

	randx = random.randint(0,w)
	randy = random.randint(0,h)
	rand = (randx,randy)

	while len(rand_list) < max_cctv: # cctv quan

		if coord_vals[randx][randy] == 1: # ERROR index out of range
			randx = random.randint(0,w)
			randy = random.randint(0,h)
			rand = (randx,randy)
		else:
			if rand in rand_list:
				randx = random.randint(0,w)
				randy = random.randint(0,h)
				rand = (randx,randy)
			else:
				# if len(rand_list) > 1:
				# 	if math.dist(rand[-2],rand[-1]) < (MIN_DISTANCE/default_scale):
				# 		randx = random.randint(0,w)
				# 		randy = random.randint(0,h)
				# 		rand = (randx,randy)
				# 	else:	
				# 		rand_list.append(rand)
				# else:	
					rand_list.append(rand)

		print(rand)
		# print(str(rand)+" : "+str(img[randx,randy][:]))
		print(coord_vals[randx][randy])
	
	print("\n ----------------------------------------------------------------")
	radius = int(CCTV_RADIUS/default_scale)

	imgArea = cv2.imread(raw_path)
	for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		imgArea = circleArea(imgArea, rand, radius, (0, 191, 0), -1)
	cv2.imwrite('imgArea.png',imgArea)

	for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		imgAreaOutline = circleArea(img, rand, radius, (0, 128, 0), 1)
	
	cv2.imwrite('imgAreaOutline.png',imgAreaOutline)
	

	imgCoord = cv2.imread(raw_path)
	for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		imgCoord = circleCoord(imgCoord, rand, 2, (191, 0, 0), -1)
		
		# print(str(rand)+" : "+str(img[randx,randy][:]))
		# print(coord_vals[rand[0]][rand[1]])

	# for rand in rand_list:
	# 	putTextCoord(img, rand, rand, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
		
	
	#---------------------------------------------------
	# img1 = cv2.imread('forest.png')
	img2 = cv2.imread(raw_path)
	dst = cv2.addWeighted(img2, 0.6, imgArea, 0.5, 0)

	# img_arr = np.hstack((img, img2))
	# cv2.imshow('Input Images',img_arr)
	# cv2.imshow('Blended Image',dst)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#---------------------------------------------------
	
	cv2.imwrite('rand.png',dst)
	print("\n")
	print(f'Updated List after removing duplicates = {rand_list}')

	return rand_list

def update_value():
	imgValue = cv2.imread('imgArea.png')
	w, h, _ = imgValue.shape

	for x in range(0,w):
		for y in range(0,h):
			if 	coord_vals[x][y] == 1:
				# walls
				imgValue[x,y] = (7, 0, 11)

			if 	(imgValue[x,y][0] < 64 and imgValue[x,y][1] > 128 and imgValue[x,y][2] < 64):
				imgValue[x,y] = (0, 191, 0)
				coord_vals[x][y] = 2
			
			if 	coord_vals[x][y] == 0:
				# walls
				imgValue[x,y] = (255, 255, 255)

	cv2.imwrite('latestValue.png',imgValue)

	raw_image = cv2.imread(raw_path)

	for x in range(0,w):
		for y in range(0,h):
			if 	coord_vals[x][y] == 1:
				# walls
				raw_image[x,y] = (7, 0, 11)

			if 	coord_vals[x][y] == 2:
				# area
				raw_image[x,y] = (0, 191, 0)
			
			if 	coord_vals[x][y] == 0:
				# available
				raw_image[x,y] = (255, 255, 255)

	cv2.imwrite('rawValue.png',raw_image)

# genetic algorithm ----------------------------------------
def cal_pop_fitness(): # value[], randList[]

	imgValue = cv2.imread('rand.png')

	for x in range(0,w):
		for y in range(0,h):
			# count total value for green area
			if 	(imgValue[x,y][0] < 64 and imgValue[x,y][1] < 64 and imgValue[x,y][2] < 128):
				# walls
				coord_vals[x][y] = 1
				imgValue[x,y] = (7, 0, 11)
			else:
				coord_vals[x][y] = 0
				imgValue[x,y] = (255,255,255)

			if 	(imgValue[x,y][0] < 128 and imgValue[x,y][1] > 128 and imgValue[x,y][2] < 128):
				# coverage
				coord_vals[x][y] = 2

    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    # fitness = np.sum(pop*equation_inputs, axis=1)

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover):
    # Mutation changes a single gene in each offspring randomly.
    for idx in range(offspring_crossover.shape[0]):
        # The random value to be added to the gene.
        random_value = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover

# ------------------------------------------------------------------------------------------------------

# driver function
if __name__=="__main__":
	# scale 1pixel:10cm  or 20pixel:6feet

	# Read floor plan image from the directory.

	# reading the image
	original_image = read_image()
	raw_path = "art.png"

	# identify image size
	# Get the width and height of the image.

	h, w, _ = original_image.shape
	# print('width: ', w)
	# print('height:', h)

	# img = area_remover(img)
	# Use cv2.selectROI() to select areas that are not available 
	# for CCTV placement (unwanted areas) and change their color to (0, 0, 64).

	availableArea_image = selectROI_area(original_image)
	

	# Use a loop for cv2.selectROI() until there are no unwanted areas left to select.
	# area_valuer() loops through all the coordinates of the image, 
	# assigning values: 0 for available areas, 1 for walls and unwanted areas.


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
	# Implement area_valuer() to assign values in a 2D array 
	# representing the coordinates in the resulting image from the selectROI() loop.

	value = area_valuer(availableArea_image)

	# print(len(value))
	# print(len(value[0]))

	# print(value[:2])

	if cv2.imwrite('after.png', availableArea_image) == True:
		print("\nafter.png saved")

	default_scale = 10

	# to initialize the CCTV quantity based on the available values and estimated total CCTV coverage.
	cctv_quantity = cctv_quantity_initializer(availableArea_image,default_scale)

	# Implement rand_coords() to draw the CCTV coverage 
	# with the color (0, 191, 0) for all generated random coordinates.
	randList = rand_coords(cctv_quantity,availableArea_image)

	# Implement update_value() to update the value for all coordinates 
	# based on the result of the rand_coords() image, comparing the color on the image with the value array.
	# Coordinates with a value of 1 will turn the pixel black, 
	# and coordinates with the color (0, 191, 0) will have a value of 2.
	# Only coordinates with a value of 0 can be changed to 2 based on the color on the image.
	update_value()

	# cal_pop_fitness()

	# print(randList[1])

	# test = [(1,1),(5,5),(10,10)]

	# print (math.dist(test[0],test[2]))

# ---------------------------------------------------------
print("\nProcessing Result ...")








"""

Current Code:

Read floor plan image from the directory.
Get the width and height of the image.
Use cv2.selectROI() to select areas that are not available for CCTV placement (unwanted areas) and change their color to (0, 0, 64).
Use a loop for cv2.selectROI() until there are no unwanted areas left to select.
Implement area_valuer() to assign values in a 2D array representing the coordinates in the resulting image from the selectROI() loop.
The size of the 2D array is based on the width and height of the image.
area_valuer() loops through all the coordinates of the image, assigning values: 0 for available areas, 1 for walls and unwanted areas.
Implement cctv_quantity_initializer() to initialize the CCTV quantity based on the available values and estimated total CCTV coverage.
The CCTV quantity is used as the number of genes in the chromosome for the genetic algorithm to generate random coordinates only on coordinates with an available value of 0.
Implement rand_coords() to draw the CCTV coverage with the color (0, 191, 0) for all generated random coordinates.
Implement update_value() to update the value for all coordinates based on the result of the rand_coords() image, comparing the color on the image with the value array.
Coordinates with a value of 1 will turn the pixel black, and coordinates with the color (0, 191, 0) will have a value of 2.
Only coordinates with a value of 0 can be changed to 2 based on the color on the image.



Ongoing Plans:

Evaluate the fitness value using the values for each pixel based on the value array.
Implement a penalty function that increases the distance between all coordinates and walls if they are below the minimum distance, to be used in the genetic algorithm pseudocode.
Update the value for overlapping areas.
Implement the functions: select_mating_pool(), crossover(), and mutation() for the initial population in the genetic algorithm.
Implement a decision-making function to evaluate the best population's fitness value and determine whether to decrease, increase, or keep the initial CCTV quantity.
If the CCTV quantity needs to be changed, run the genetic algorithm again to find the best population with the new generated CCTV quantity.
Use a feed-forward or feed-backward technique in the decision-making process.


"""