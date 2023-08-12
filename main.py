# from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import cv2
import math
# from matplotlib import pyplot as plt
import pandas as pd
import sys
import random
import numpy as np
from array import *
import operator as op
import os

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# declaration
raw_path = "art.png"
# raw_path = "parking.png"
coords = []
value = []
default_scale = 10

img = cv2.imread(raw_path, 1)
w, h, c = img.shape
W = w-1
H = h-1

coordVals = [[0 for y in range(0,H)] for x in range(0,W)]

CCTV_RADIUS = 1000
MIN_DISTANCE = 900
MAX_DISTANCE = 1100

CCTV_DIST_WEIGHT = 1.333

WALL_PENALTY = 0.333
OVERLAP_PENALTY = 0.666

if os.path.exists("RESULT.txt"):
	os.remove("RESULT.txt")
if os.path.exists("FITNESS.txt"):
	os.remove("FITNESS.txt")
if os.path.exists("AVERAGE.txt"):
	os.remove("AVERAGE.txt")

# ------------------------------------------------------------

# Python implementation to
# read last N lines of a file

# Function to read
# last N lines of the file
def LastNlines(fname):
	# opening file using with() method
	# so that file get closed
	# after completing work
    with open(fname) as file:
        for line in file:
            pass
        last_line = line

        # print(last_line)

    return last_line

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

# calculate slope between two coordinates to increase distance between them N/A
def slope(x1, y1, x2, y2):
    if(x2 - x1 != 0):
      return (float)(y2-y1)/(x2-x1)
    return sys.maxint

# penalty for GA if distance between coordinates less than minimum distance N/A
def coord_penalty(img, coord):
	# w,h,_=img.shape
	penalty = True

	for x in range(0,W):
		for y in range(0,H):
			if 	coordVals[x][y] == 1:
				while (penalty == True):
					if math.dist((int(x),int(y)),coord) < (MIN_DISTANCE/default_scale):
						if int(x) < coord[0] and int(y) < coord[1]:
							coord = ((coord[0]+1),(coord[1]+1))

						if int(x) < coord[0] and int(y) > coord[1]:
							coord = ((coord[0]+1),(coord[1]-1))

						if int(x) > coord[0] and int(y) < coord[1]:
							coord = ((coord[0]-1),(coord[1]+1))

						if int(x) > coord[0] and int(y) > coord[1]:
							coord = ((coord[0]-1),(coord[1]-1))
					else:
						penalty = False

					# coord_vals[x][y] = 1
					# img[x,y] = (7, 0, 11)

# Python3 program to check if a point lies inside a circle or not
def isInside(circle_x, circle_y, rad, x, y):
     
    # Compare radius of circle
    # with distance of its center
    # from given point
    if ((x - circle_x) * (x - circle_x) +
        (y - circle_y) * (y - circle_y) <= rad * rad):
        return True;
    else:
        return False;

#function which returns the index of minimum value in the list N/A
def get_minidx(inputlist):
 
    #get the minimum value in the list
    min_value = min(inputlist)
 
    #return the index of minimum value 
    min_index=inputlist.index(min_value)
    return min_index

# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# read floor plan image -------------------------
def read_image():
	img = cv2.imread(raw_path, 1)
	return img

# preprocess image ------------------------- ???
def binarization(): # NOT USING
	img = read_image()
	# w, h, _ = img.shape
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
	
	# cv2.imwrite("bw.png", img)

# obtain coordinate for polygon -------------------------
def draw_rectangle(image,coords): # N/A
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

def click_event(event, x, y, flags, params): # N/A
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
def area_remover(img): # N/A
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
def areaValuer():

	availableImage = cv2.imread('result/availableArea.png',1)

	for x in range(0,W):
		for y in range(0,H):
			# existed walls
			if (availableImage[x,y][0] < 64 and availableImage[x,y][1] < 64 and availableImage[x,y][2] < 64):
				coordVals[x][y] = 1
				availableImage[x,y] = (7, 0, 11) # neutral black

			# unwanted areas
			if (availableImage[x,y][0] < 64 and availableImage[x,y][1] < 64 and availableImage[x,y][2] < 128):
				coordVals[x][y] = 1
				availableImage[x,y] = (7, 0, 11) # neutral black

			# avl areas
			else:
				coordVals[x][y] = 0
				availableImage[x,y] = (255,255,255) # pure white

	valuedImage = cv2.imwrite('result/bw.png',availableImage)

	if pd.DataFrame(coordVals).to_csv('coordVals.csv')==True: 
		print("coordVals.csv saved")

	return valuedImage

# def selectROI_area(image):
# 	r = cv2.selectROI("select the area", image)

# 	image = cv2.rectangle(image, (int(r[0]),int(r[1])), 
# 		(int(r[0]+r[2]),int(r[1]+r[3])), (0,0,64), -1)

# 	if cv2.imwrite('result/availableArea.png', image) == True:
# 		print("\navailableArea.png updated")

# 	return image

def selectROI_area(image, output_filename='result/availableArea.png', mark_color=(0, 0, 64)):
    r = cv2.selectROI("select the area", image)
    image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), mark_color, -1)

    if cv2.imwrite(output_filename, image):
        print(f"\n{output_filename} updated")

    return image

# initialize cctv number based on area? -------------------------
def initializeSol(scale,weightCCTV):
	wall = 0
	qty = 0

	for x in range(0,W):
		for y in range(0,H):
			if 	coordVals[x][y] == 1: # wall and unwanted area
				# WALL
				wall += 1

	clear_area = (h * w) - wall

	qty = round(clear_area / (weightCCTV*(math.pi * math.pow((CCTV_RADIUS/scale),2))))
	
	print("\nInitializing cctv quantity: ", qty, "\n")
	return qty

def randomizer():
	randx = random.randint(10,W)
	randy = random.randint(10,H)
	rand = (randx,randy)
	return randx, randy, rand

def randCoords(index,max_cctv):

	imgOutline = cv2.imread(raw_path,1)

	# rand_val = [[0 for y in range(0,H)] for x in range(0,W)]

	rand_list = []

	randx, randy, rand = randomizer()

	# generate genes

	while len(rand_list) < max_cctv: # cctv quan

		if randx < W and randy < H:

			if coordVals[randx][randy] != 1:

				if rand not in rand_list:
					
					if 	(imgOutline[randx,randy][0] < 192 and imgOutline[randx,randy][1] < 192 and imgOutline[randx,randy][2] < 192):
						randx, randy, rand = randomizer()
					
					else:
						rand_list.append(rand)
						radius = int(CCTV_DIST_WEIGHT*CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgOutline = circleArea(imgOutline, rand, radius, (0, 191, 0), -1)
					
				else:
					randx, randy, rand = randomizer()

			else:
				randx, randy, rand = randomizer()
				
		else:
			randx, randy, rand = randomizer()
	
	print(("... Solution Candidate "+ str(index+1) +" Generated ...\n"))

	radius = int(CCTV_RADIUS/default_scale)

	# imgOutline = cv2.imread(raw_path,1)
	# i = 1
	# for rand in rand_list:

	# 	# rand[1],rand[0] for image !!!!!!!!!!!
	# 	imgOutline = circleArea(imgOutline, rand, radius, (0, 128, 0), 1)
	# 	imgOutline = circleCoord(imgOutline, rand, 2, (191, 0, 0), -1)
	# 	cv2.putText(imgOutline, str(i), (rand[1],rand[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
	# 	i += 1

	# cv2.imwrite(('result/imgAreaOutline' + str(index) + '.png'),imgOutline)
	

	imgCoord = cv2.imread(raw_path)
	for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		imgCoord = circleCoord(imgCoord, rand, 2, (191, 0, 0), -1)
		
		# print(str(rand)+" : "+str(img[randx,randy][:]))
		# print(coord_vals[rand[0]][rand[1]])

	return rand_list


# def randCoords(index, max_cctv, raw_path, coordVals, CCTV_RADIUS, default_scale):
    
#     imgOutline = cv2.imread(raw_path, 1)
#     # W, H = imgOutline.shape[0], imgOutline.shape[1]

#     rand_list = []

#     while len(rand_list) <= max_cctv:
#         randx = random.randint(0, W)
#         randy = random.randint(0, H)
#         rand = (randx, randy)

#         if randx < W and randy < H and coordVals[randx][randy] != 1:
#             if rand not in rand_list and (
#                     imgOutline[randx, randy][0] >= 192 or imgOutline[randx, randy][1] >= 192 or
#                     imgOutline[randx, randy][2] >= 192
#             ):
#                 rand_list.append(rand)
#                 radius = int(1.5 * CCTV_RADIUS / default_scale)
#                 imgOutline = circleArea(imgOutline, rand, radius, (0, 191, 0), -1)

#         # Generate new random coordinates if constraints are not met
#         randx = random.randint(0, W)
#         randy = random.randint(0, H)

#     print(("Gene "+ str(index) +" Generated ...\n"))

#     radius = int(CCTV_RADIUS / default_scale)

#     imgArea = cv2.imread(raw_path)
#     for rand in rand_list:
#         imgArea = circleArea(imgArea, rand, radius, (0, 191, 0), -1)

#     cv2.imwrite(('result/imgArea' + str(index) + '.png'), imgArea)

#     imgOutline = cv2.imread(raw_path)
#     i = 1
#     for rand in rand_list:
#         imgOutline = circleArea(imgOutline, rand, radius, (0, 128, 0), 1)
#         imgOutline = circleCoord(imgOutline, rand, 2, (191, 0, 0), -1)
#         cv2.putText(imgOutline, str(i), (rand[1], rand[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
#         i += 1

#     cv2.imwrite(('result/imgAreaOutline' + str(index) + '.png'), imgOutline)

#     imgCoord = cv2.imread(raw_path)
#     for rand in rand_list:
#         imgCoord = circleCoord(imgCoord, rand, 2, (191, 0, 0), -1)

#     img2 = cv2.imread('bw.png', 1)
#     dst = cv2.addWeighted(img2, 0.6, imgArea, 0.6, 0)

#     cv2.imwrite('result/rand' + str(index) + '.png', dst)

#     return rand_list


def chrValue(index,chrList):
	
	avlImg = cv2.imread('result/bw.png',1)
	radius = int(CCTV_RADIUS/default_scale)
	for chr in chrList:
		# rand[1],rand[0] for image !!!!!!!!!!!
		avlImg = circleArea(avlImg, chr, radius, (0, 191, 0), -1)
	
	image_path = ('result/imgArea') + str(index) + '.png'
	img = cv2.imread(image_path,1)

	for x in range(0,W):
		for y in range(0,H):
			if 	coordVals[x][y] == 1:
				# walls
				img[x,y] = (7, 0, 11)

			if 	(img[x,y][0] < 64 and img[x,y][1] > 128 and img[x,y][2] < 64):
				img[x,y] = (0, 191, 0)
				coordVals[x][y] = 2
				# print(2)
			
			if 	coordVals[x][y] == 0:
				# not walls
				img[x,y] = (255, 255, 255)

	image_path = ('result/imgValue') + str(index) + '.png'
	cv2.imwrite(image_path,img)

	# raw_image = cv2.imread(raw_path)

	# for x in range(0,W):
	# 	for y in range(0,H):
	# 		if 	coordVals[x][y] == 1:
	# 			# walls
	# 			raw_image[x,y] = (7, 0, 11)

	# 		if 	coordVals[x][y] == 2:
	# 			# area
	# 			raw_image[x,y] = (0, 191, 0)
			
	# 		if 	coordVals[x][y] == 0:
	# 			# available
	# 			raw_image[x,y] = (255, 255, 255)

	# # cv2.imwrite('rawValue.png',raw_image)
	# image_path = ('result/rawValue') + str(index) + '.png'
	# cv2.imwrite(image_path,raw_image)

# genetic algorithm ----------------------------------------
# def initialize_population(pop_size, max_cctv, W, H):
#     population = []
#     for _ in range(pop_size):
#         individual = []
#         while len(individual) < max_cctv:
#             randx = random.randint(50, W)
#             randy = random.randint(50, H)
#             gene = (randx, randy)
#             if is_valid_gene(gene, individual):
#                 individual.append(gene)
#         population.append(individual)
#     return population

def cal_pop_fitness(index,chr): # value[], randList[]

	# find total white pixel

	available_path = 'result/availableArea.png'
	imgAvailable = cv2.imread(available_path,1)

	totalAvailableArea = 0

	for x in range(0,W):
		for y in range(0,H):
			if (imgAvailable[x,y][0] > 250 and imgAvailable[x,y][1] > 250 and imgAvailable[x,y][2] > 250):
			# if coordVals[x][y] == 1:
				totalAvailableArea += 1

	# print("available : "+str(totalAvailableArea))

	fitness = 0
	# covered_coordinates = 0
	# coverage_percentage = 0.0
	# unwanted_penalty = 0
	# coverable = 0
	# total_coordinates = 0
	gene_val = [[0 for y in range(0,H)] for x in range(0,W)]
	gene_fitness = []

	worst_gene_fitness = 999
	worst_gene_idx = 0


	radius = int(CCTV_RADIUS/default_scale)
	imgArea = cv2.imread(raw_path)
	imgOutline = cv2.imread(raw_path)
	i=1

	for gene in chr:
		
		imgArea = circleArea(imgArea, gene, radius, (0, 191, 0), -1)
		imgOutline = circleArea(imgOutline, gene, radius, (0, 128, 0), 1)
		imgOutline = circleCoord(imgOutline, gene, 2, (191, 0, 0), -1)
		cv2.putText(imgOutline, str(i), (gene[1],gene[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
		i += 1

		# Driver Code
		# x = 1
		# y = 1
		circle_x = gene[0]
		circle_y = gene[1]
		rad = int(CCTV_RADIUS/default_scale)
		total_gene_val = 0
		gene_penalty = 0

		for x in range(0,W):
			for y in range(0,H):

				if(isInside(circle_x, circle_y, rad, x, y)):
					# print("Inside")
					# area cover not wall
					if gene_val[x][y] == 0 and coordVals[x][y] != 1:
						gene_val[x][y] = 1
						total_gene_val += gene_val[x][y]

					# area cover wall
					if gene_val[x][y] == 0 and coordVals[x][y] == 1:
						gene_penalty = op.add(gene_penalty,WALL_PENALTY)
						# imgArea[x,y] = (7, 0, 11) # neutral black
						imgArea[x,y] = (127,127,0) # cyan 
						imgOutline[x,y] = (127,0,127) # 

					# area overlap other area
					if gene_val[x][y] > 0:
						gene_val[x][y] += 1
						gene_penalty = op.add(gene_penalty,OVERLAP_PENALTY)

					if gene_val[x][y] > 2:
						imgArea[x,y] = (127,127,0)
						imgOutline[x,y] = (127,127,0)

					# if coordVals[x][y] == 1:
					# 	gene_penalty = op.add(gene_penalty,1)
					# else:
					# 	total_gene_val = op.add(total_gene_val,1)

		total_gene_val = total_gene_val - gene_penalty
		gene_fitness.append(int(total_gene_val))

		if worst_gene_fitness < total_gene_val:
			worst_gene_fitness = total_gene_val
			worst_gene_idx = i-1

	cv2.imwrite(('result/imgArea' + str(index) + '.png'),imgArea)
	cv2.imwrite(('result/imgAreaOutline' + str(index) + '.png'),imgOutline)

	# ----- ----- ----- ----- -----

	img2 = cv2.imread('result/bw.png',1)
	dst = cv2.addWeighted(img2, 0.6, imgArea, 0.6,0)
	
	cv2.imwrite('result/imgOverlay'+str(index)+'.png',dst) # double exposure image

	# ----- ----- ----- ----- -----

	# print(f"gene_fitness: ",gene_fitness)

	# NORMAL METHOD
	# for x in range(0,W):
	# 	for y in range(0,H):
		
	# 		# if dark green value 1, penalty wall
	# 		if 	(imgValue[x,y][0] < 64 and imgValue[x,y][1] > 64 and imgValue[x,y][2] < 64):
	# 			unwanted_penalty = op.add(unwanted_penalty,1)
	# 			# print(1)

	# 		# if light green value 0, value set 2 and covered
	# 		if 	(imgValue[x,y][0] < 192 and imgValue[x,y][1] > 192 and imgValue[x,y][2] < 192):
	# 		# if 	(imgValue[x,y][0] < 64 and imgValue[x,y][1] > 128 and imgValue[x,y][2] < 64):

	# 			# coverage
	# 			# coordVals[x][y] = 2
	# 			covered_coordinates = op.add(covered_coordinates,1)
	# 			total_coordinates = op.add(total_coordinates,1)
	# 			# print(2)

	# 		if coordVals[x][y] != 1:
	# 			# coordVals[x][y] = 0
	# 			# imgValue[x,y] = (255,255,255)
	# 			total_coordinates = op.add(total_coordinates,1)
	# 			coverable += 1
	# 			# print(0)
				

	# method 1
	# if dark green value 1, penalty wall
	# if light green value 0, value set 2 and covered
	# coverage_percentage = covered_coordinates / totalAvailableArea
	# fitness = coverage_percentage - (unwanted_penalty/total_coordinates)
	
	# method 2
	fitness = sum(gene_fitness) / totalAvailableArea

	# print(covered_coordinates)
	# print(total_coordinates)
	# print(unwanted_penalty)
	# print("Formula 1 fitness: ",fitness)
	# print("Formula 2 fitness: ",fitnessg)

	# print("\nfitness value calculated ...")
    
	return fitness, worst_gene_fitness, worst_gene_idx
	# return fitnessg
    # Calculating the fitness value of each solution in the current population.

def selection(population, fitness_values):
    # Perform selection based on worst gene fitness value each solution
    # to create a new population for the next generation
    # use roulette wheel selection
	li=[]
 
	for i in range(len(fitness_values)):
		li.append([fitness_values[i],i])
	
	# ascending order smol to beeg
	li.sort()
	sort_index = []
	
	for x in li:
		sort_index.append(x[1])
	
	# print(sort_index)

	
	last_two_indices = sort_index[:2]
	first_two_indices = sort_index[-2:]
	indices = last_two_indices + first_two_indices
	# print(indices)

    # return last_two_indices
	return indices

def crossover(parents, offspring_size):
	# offspring = [0]*offspring_size
	offspring = []
	crossover_point = int(offspring_size/2) # index 6

	p1 = parents[0]
	p2 = parents[1]

	cd1 = p1[0:crossover_point]
	cd2 = p2[0:crossover_point]

	j=crossover_point # 7
	for i in range(offspring_size-crossover_point):
		cd1.append(p2[int(i+j)])
		cd2.append(p1[int(i+j)])

	offspring.append(cd1)
	offspring.append(cd2)

	# print("p1: ",p1)
	# print("p2: ",p2,"\n")
	# print("o1: ",offspring[0])
	# print("o2: ",offspring[1])
	return offspring

def mutation(pop, parent_idx, worst_sol_idx):

	# offspring = []
	mutated_offspring = []
	# offspring_idx = []
	# gene = []
	# imgArea = []

	# create offspring using parent idx

	offspring1 = pop[parent_idx[0]]
	offspring2 = pop[parent_idx[1]]
	offspring3 = pop[parent_idx[2]]
	offspring4 = pop[parent_idx[3]]

	# for i in range(4):
	# 	offspring.append(pop[parent_idx[i]])
	# 	offspring_idx.append(parent_idx[i])
	# 	gene.append(offspring[i][offspring_idx[i]])
		# imgArea[i] = cv2.imread(available_path)

	# print("\n")
	# print(parent_idx)

	# print(offspring1)
	# print(offspring2)

	# worst gene idx in each offspring

	offspring1_idx = parent_idx[0]
	offspring2_idx = parent_idx[1]
	offspring3_idx = parent_idx[2]
	offspring4_idx = parent_idx[3]

	gene1 = offspring1[offspring1_idx]
	gene2 = offspring2[offspring2_idx]
	gene3 = offspring3[offspring3_idx]
	gene4 = offspring4[offspring4_idx]

	# print(worst1_idx)
	# print(worst2_idx)

	# print(worst_sol_idx)
	
	available_path = 'result/availableArea.png'

	imgArea1 = cv2.imread(available_path)
	imgArea2 = cv2.imread(available_path)
	imgArea3 = cv2.imread(available_path)
	imgArea4 = cv2.imread(available_path)

	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0

	radius = int(CCTV_RADIUS/default_scale)

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	# for i in range(len(count)):
	# 	imgArea = cv2.imread(available_path)

	# 	for gene in offspring[i]:
	# 		if gene != gene[i]:
	# 			imgArea = circleArea(imgArea, gene, radius, (0, 191, 0), -1)
	# 			count[i] += 1

	# 	while len(offspring[i]) > count[i]:
	# 		if randx < W and randy < H:

	# 			if coordVals[randx][randy] != 1:

	# 				if rand not in offspring[i]:
						
	# 					if 	(imgArea[randx,randy][0] < 192 and imgArea[randx,randy][1] < 192 and imgArea[randx,randy][2] < 192):
	# 						randx, randy, rand = randomizer()
						
	# 					else:
	# 						offspring[i][offspring_idx[i]] = rand
	# 						print(str(gene[i])+" >>> "+str(rand))
	# 						radius = int(CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
	# 						imgArea = circleArea(imgArea, rand, radius, (0, 191, 0), -1)
	# 						count[i] += 1
	# 						mutated_offspring.append(offspring[i])
						
	# 				else:
	# 					randx, randy, rand = randomizer()

	# 			else:
	# 				randx, randy, rand = randomizer()
					
	# 		else:
	# 			randx, randy, rand = randomizer()

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	for gene in offspring1:
		if gene != gene1:
			imgArea1 = circleArea(imgArea1, gene, radius, (0, 191, 0), -1)
			count1 += 1

	for gene in offspring2:
		if gene != gene2:
			imgArea2 = circleArea(imgArea2, gene, radius, (0, 191, 0), -1)
			count2 += 1

	for gene in offspring3:
		if gene != gene3:
			imgArea1 = circleArea(imgArea3, gene, radius, (0, 191, 0), -1)
			count3 += 1

	for gene in offspring4:
		if gene != gene4:
			imgArea2 = circleArea(imgArea4, gene, radius, (0, 191, 0), -1)
			count4 += 1

	# print(count1)
	# print(count2)

	randx, randy, rand = randomizer()

	while len(offspring1) > count1:
		if randx < W and randy < H:

			if coordVals[randx][randy] != 1:

				if rand not in offspring1:
					
					if 	(imgArea1[randx,randy][0] < 192 and imgArea1[randx,randy][1] < 192 and imgArea1[randx,randy][2] < 192):
						randx, randy, rand = randomizer()
					
					else:
						offspring1[offspring1_idx] = rand
						print(str(gene1)+" >>> "+str(rand))
						radius = int(CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgArea1 = circleArea(imgArea1, rand, radius, (0, 191, 0), -1)
						count1 += 1
					
				else:
					randx, randy, rand = randomizer()

			else:
				randx, randy, rand = randomizer()
				
		else:
			randx, randy, rand = randomizer()

	randx, randy, rand = randomizer()

	while len(offspring2) > count2:
		if randx < W and randy < H:

			if coordVals[randx][randy] != 1:

				if rand not in offspring2:
					
					if 	(imgArea2[randx,randy][0] < 192 and imgArea2[randx,randy][1] < 192 and imgArea2[randx,randy][2] < 192):
						randx, randy, rand = randomizer()
					
					else:
						offspring2[offspring2_idx] = rand
						print(str(gene2)+" >>> "+str(rand))
						radius = int(CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgArea2 = circleArea(imgArea2, rand, radius, (0, 191, 0), -1)
						count2 += 1
					
				else:
					randx, randy, rand = randomizer()

			else:
				randx, randy, rand = randomizer()
				
		else:
			randx, randy, rand = randomizer()

	randx, randy, rand = randomizer()

	while len(offspring3) > count3:
		if randx < W and randy < H:

			if coordVals[randx][randy] != 1:

				if rand not in offspring3:
					
					if 	(imgArea3[randx,randy][0] < 192 and imgArea3[randx,randy][1] < 192 and imgArea3[randx,randy][2] < 192):
						randx, randy, rand = randomizer()
					
					else:
						offspring3[offspring3_idx] = rand
						print(str(gene3)+" >>> "+str(rand))
						radius = int(CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgArea3 = circleArea(imgArea3, rand, radius, (0, 191, 0), -1)
						count3 += 1
					
				else:
					randx, randy, rand = randomizer()

			else:
				randx, randy, rand = randomizer()
				
		else:
			randx, randy, rand = randomizer()

	randx, randy, rand = randomizer()

	while len(offspring4) > count4:
		if randx < W and randy < H:

			if coordVals[randx][randy] != 1:

				if rand not in offspring4:
					
					if 	(imgArea4[randx,randy][0] < 192 and imgArea4[randx,randy][1] < 192 and imgArea4[randx,randy][2] < 192):
						randx, randy, rand = randomizer()
					
					else:
						offspring4[offspring4_idx] = rand
						print(str(gene4)+" >>> "+str(rand))
						radius = int(CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgArea2 = circleArea(imgArea4, rand, radius, (0, 191, 0), -1)
						count4 += 1
					
				else:
					randx, randy, rand = randomizer()

			else:
				randx, randy, rand = randomizer()
				
		else:
			randx, randy, rand = randomizer()

	# print(offspring1)
	# print(offspring2)

	mutated_offspring.append(offspring1)
	mutated_offspring.append(offspring2)
	mutated_offspring.append(offspring3)
	mutated_offspring.append(offspring4)

	# print(offspring)

	# exit()

	# for chr in pop:

	# 	gene_val = [[0 for y in range(0,H)] for x in range(0,W)]
	# 	gene_fitness = []
	# 	for gene in chr:
	# 		circle_x = gene[0]
	# 		circle_y = gene[1]
	# 		rad = int(CCTV_RADIUS/default_scale)
	# 		total_gene_val = 0
	# 		gene_penalty = 0

			

	# 		for x in range(0,W):
	# 			for y in range(0,H):

	# 				if(isInside(circle_x, circle_y, rad, x, y)):
	# 					# print("Inside")
	# 					# area cover not wall
	# 					if gene_val[x][y] == 0 and coordVals[x][y] != 1:
	# 						gene_val[x][y] = 1
	# 						total_gene_val += gene_val[x][y]

	# 					# area cover wall
	# 					if gene_val[x][y] == 0 and coordVals[x][y] == 1:
	# 						gene_penalty = op.add(gene_penalty,0.3)

	# 					# area overlap
	# 					if gene_val[x][y] > 0:
	# 						gene_val[x][y] += 1
	# 						gene_penalty = op.add(gene_penalty,0.3)

	# 		total_gene_val = total_gene_val - gene_penalty
	# 		gene_fitness.append(int(total_gene_val))

	# 	# mutation, remove least value coord
	# 	min_val = min(gene_fitness)
	# 	min_fitness_idx = gene_fitness.index(min_val)

	# 	# print(chr[min_fitness_idx]," : ",min_fitness_idx," : ",min_val)

	# 	# print(gene_fitness)

	# 	chr.pop(min_fitness_idx)

	# 	x = random.randint(50,W)
	# 	y = random.randint(50,H)
	# 	rand = (x,y)

	# 	while (gene_val[x][y] != 0):
			
	# 		x = random.randint(50,W)
	# 		y = random.randint(50,H)
	# 		rand = (x,y)

	# 	# mutation, replace with new coord
	# 	chr.append(rand)

	# print(pop[0])
	# print(pop[1])

	return mutated_offspring

# ------------------------------------------------------------------------------------------------------

# driver function
if __name__=="__main__":
	# scale = 10 , 1pixel:10cm  or 20pixel:6feet

	# Read floor plan image from the directory.
	original_image = read_image()
	# raw_path = "art.png"

	print("Image Width  : "+str(W))
	print("Image Height : "+str(H))

	# Use cv2.selectROI() to select areas that are not available 
	# for CCTV placement (unwanted areas) and change their color to (0, 0, 64) red.

	# Use a loop for cv2.selectROI() until there are no unwanted areas left to select.
	# area_valuer() loops through all the coordinates of the image, 
	# assigning values: 0 for available areas, 1 for walls and unwanted areas.

	availableArea_image = selectROI_area(original_image)

	stop = False
	while stop == False:
		print("\nPress 's' to stop")

		k = cv2.waitKey(0)
		if k == ord("s") or k == ord("S"):
			stop = True
			print("\nFinalizing available area ...")
		else:
			availableArea_image = selectROI_area(availableArea_image)

	cv2.destroyAllWindows()

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	# PARAMETER SETTINGS
	convergence_threshold = 0.001

	# assign value to coord
	# 1 = wall, remove
	# 0 = empty
	# validation only
	# Implement area_valuer() to assign values in a 2D array 
	# representing the coordinates in the resulting image from the selectROI() loop.

	valuedImage  = areaValuer()

	# to initialize the CCTV quantity (solution) based on the available values and estimated total CCTV coverage.
	chr_size = initializeSol(default_scale, weightCCTV=1.0)

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	# Implement rand_coords() to draw the CCTV coverage 
	# with the color (0, 191, 0) for all generated random coordinates.
	
	# print(f'Updated List after removing duplicates = {new_population}')

	# Implement update_value() to update the value for all coordinates 
	# based on the result of the rand_coords() image, comparing the color on the image with the value array.
	# Coordinates with a value of 1 will turn the pixel black, 
	# and coordinates with the color (0, 191, 0) will have a value of 2.
	# Only coordinates with a value of 0 can be changed to 2 based on the color on the image.
	
	# update_value()

	# ---------------------------------------------------------

	initial_population = []
	fitness = []
	
	sol_per_pop = 8
	num_parents_mating = 2
	
	num_generations = 33

	previous_fitness = 0.000
	best_chr_fitness = -999
	best_sol_fitness = -999
	best_sol_idx = 0
	best_sol = []

	# Measing the fitness of each chromosome in the population.
	for x in range(sol_per_pop):
		chr = randCoords(x, chr_size)
		initial_population.append(chr)

		# chrFitness = cal_pop_fitness(x,new_population[x])
		# fitness.append(chrFitness)

		chrValue(x,chr)

	print("\n... Initial Population ...\n")
	print(initial_population)

	current_population = initial_population
	
	for generation in range(num_generations):
		print("\n... Generation : ", generation+1," ...\n")

		print("\n... Current Population ...\n")
		print(current_population)

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		print("\n... Fitness Function ...\n")

		worst_sol_fitness = [] 	# worst gene fitness each sol
		worst_sol_idx = []		# worst gene idx each sol

		worst_gene_fitness = 0
		worst_gene_idx = 0
	
		# # Measing the fitness of each chromosome in the solution.
		for x in range(sol_per_pop):
			chrFitness, worst_gene_fitness, worst_gene_idx = cal_pop_fitness(x,current_population[x])
			
			fitness.append(chrFitness)

			# selection, crossover, mutation variables for GENES
			worst_sol_fitness.append(worst_gene_fitness)
			worst_sol_idx.append(worst_gene_idx)

			print("\nfitness "+str(x)+" : "+str(chrFitness)+"\tworst gene "+str(worst_gene_idx)+" : "+str(current_population[x][worst_gene_idx])+"\n")
			# print("worst fitness "+str(x)+" : "+str(worst_gene_fitness)+"\n")

			if chrFitness > best_chr_fitness:
				best_chr_fitness = chrFitness
				best_chr_idx = x
				best_chr = current_population[best_chr_idx]

			chrValue(x,chr)

		average_fitness = sum(fitness) / len(fitness)
		
		print("\nBest result : ",best_chr)
		print("Best fitness : ",best_chr_fitness)
		print("Average fitness : ",average_fitness,"\n")

		# save best sol every gen
		best_gen_sol = 'result/imgAreaOutline' + str(best_chr_idx) + '.png'
		imgBestGenSol = cv2.imread(best_gen_sol,1)
		# best_gen_sol_path = 'result/imgBestSolGen' + str(generation) + str(best_chr_idx) + '.png'
		best_gen_sol_path = 'result/test_result.png'
		cv2.imwrite(best_gen_sol_path, imgBestGenSol)
		# cv2.imshow('result',bestSolImg)

		# update result, fitness, average

		# retrieve previous gen best fitness
		if os.path.exists("FITNESS.txt"):
			fname = 'FITNESS.txt'
			previous_fitness = float(LastNlines(fname))

		# write result, fitness, average to text file PER GENERATION
		bestResult = open("RESULT.txt", "a")
		bestResult.write(str(best_chr)+"\n")
		bestResult.close()

		bestFitness = open("FITNESS.txt", "a")
		bestFitness.write(str(best_chr_fitness)+"\n")
		bestFitness.close()

		avgFitness = open("AVERAGE.txt", "a")
		avgFitness.write(str(average_fitness)+"\n")
		avgFitness.close()

		print("\n... Text file updated ...\n")

		if best_chr_fitness > best_sol_fitness:
			best_sol_fitness = best_chr_fitness
			best_sol_idx = best_chr_idx
			best_sol = best_chr

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# TERMINATION CRITERION

		# convergence = abs(best_chr_fitness - previous_fitness)
		convergence = 0

		if (convergence > 0 and convergence < convergence_threshold):

			print("... Convergence Threshold Reached "+ str(convergence_threshold) +" ...")

			# Getting the best solution after iterating finishing all generations.
			print("\nBest solution : ",best_sol)
			print("Fitness : ",best_sol_fitness)

			radius = int(CCTV_RADIUS/default_scale)
			i=1
			imgBestSol = cv2.imread(raw_path,1)
			for gene in best_sol:
				imgBestSol = circleArea(imgBestSol, gene, radius, (0, 128, 0), 1)
				imgBestSol = circleCoord(imgBestSol, gene, 2, (191, 0, 0), -1)
				cv2.putText(imgBestSol, str(gene[1])+","+str(gene[0]), (gene[1],gene[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
				i += 1

			best_image_path = 'result/imgBestSolution.png'
			cv2.imwrite(best_image_path,imgBestSol)
			# best_image = cv2.imread(best_image_path,1)
			cv2.imshow('result',imgBestSol)

			k = cv2.waitKey(0)
			if k == ord("s") or k == ord("S"):
				cv2.destroyAllWindows
			print("\n... System Terminated ...\n\n")
			exit()

		print("\nworst gene fitness : ",worst_sol_fitness)
		print("worst gene idx : ",worst_sol_idx,"\n")

		# print(get_minidx(worst_sol_fitness))

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		print("\n... Parent Selection ...\n")
		
		# Selecting the best parents in the population for mating. parents = (sol idx in pop)
		parents = selection(current_population, worst_sol_fitness)
		# print(parents)
		print("\nParent 1 : "+str(current_population[parents[0]]))
		print("\nParent 2 : "+str(current_population[parents[1]]))
		print("\nParent 3 : "+str(current_population[parents[2]]))
		print("\nParent 4 : "+str(current_population[parents[3]]))

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# Generating next generation using crossover.
		# No crossover. Copy parents to offsprings
		# offsprings = parents

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		print("\n... Offspring Mutation ...\n")

		# Adding some variations to the offsrping using mutation.
		mutated_offsprings = mutation(current_population, parents, worst_sol_idx)

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# Creating the new population based on the parents and offspring.

		current_population[parents[0]] = mutated_offsprings[0]
		current_population[parents[1]] = mutated_offsprings[1]
		current_population[parents[2]] = mutated_offsprings[2]
		current_population[parents[3]] = mutated_offsprings[3]

		print("\n... Survival Population ...\n")
		print(current_population)

		fitness.clear()
		worst_sol_fitness.clear()
		worst_sol_idx.clear()

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# generation end

# ---------------------------------------------------------

k = cv2.waitKey(0)
if k == ord("s") or k == ord("S"):
	cv2.destroyAllWindows
print("\n... System Terminated ...\n\n")








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