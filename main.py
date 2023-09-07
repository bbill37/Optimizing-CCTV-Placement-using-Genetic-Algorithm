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
import argparse
import subprocess


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Create an argument parser
parser = argparse.ArgumentParser(description="CCTV Placement Optimization")

# Add arguments for floor plan and weightage
parser.add_argument("--floorplan", type=str, help="Path to the floor plan image", required=True)
parser.add_argument("--weightage", type=float, help="Weightage value", required=True)

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
floorplan_path = args.floorplan
weightage_value = float(args.weightage)

# Now you can use floorplan_path and weightage_value in your optimization algorithm
print(f"Floor Plan Path: {floorplan_path}")
print(f"Weightage Value: {weightage_value}")



# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# declaration
raw_path = floorplan_path
# raw_path = "parking.png"
coords = []
value = []
default_scale = 10

img = cv2.imread(raw_path, 1)
w, h, c = img.shape
W = w
H = h

coordVals = [[0 for y in range(0,H)] for x in range(0,W)]

CCTV_RADIUS = 600
MIN_DISTANCE = 900
MAX_DISTANCE = 1100

NUM_GENERATION = 100

CCTV_DIST_WEIGHT = weightage_value # cctv rand dist !!!!! CHAPTER 6: TESTING
weightCCTV = 1.0 # cctv qty
WALL_PENALTY = -1.0 # fitness
OVERLAP_PENALTY = 0.5 # fitness
WHITE_RANGE_WEIGHT = 1.0 # mutation

if os.path.exists("RESULT.txt"):
	os.remove("RESULT.txt")
if os.path.exists("FITNESS.txt"):
	os.remove("FITNESS.txt")
if os.path.exists("FITTEST.txt"):
	os.remove("FITTEST.txt")
if os.path.exists("AVERAGE.txt"):
	os.remove("AVERAGE.txt")
if os.path.exists("PARAMETER.txt"):
	os.remove("PARAMETER.txt")

parameterSetting = open("PARAMETER.txt", "a")
parameterSetting.write(str(CCTV_DIST_WEIGHT)+" CCTV_DIST_WEIGHT\n")
parameterSetting.write(str(weightCCTV)+" weightCCTV\n")
parameterSetting.write(str(NUM_GENERATION)+" NUM_GENERATION\n")
parameterSetting.write(str(WALL_PENALTY)+" WALL_PENALTY\n")
parameterSetting.write(str(OVERLAP_PENALTY)+" OVERLAP_PENALTY\n")
parameterSetting.write(str(WHITE_RANGE_WEIGHT)+" WHITE_RANGE_WEIGHT\n")
parameterSetting.close()

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

def calculateWhite(circle_x, circle_y, path, randx, randy):

	imgArea = cv2.imread(path,1)
	rad = int(CCTV_RADIUS/default_scale)
	white_val = 0
	
	for x in range((W)):
		for y in range((H)):

			if(isInside(circle_x, circle_y, rad, x, y)):
				if (imgArea[randx,randy][0] > 224 and imgArea[randx,randy][1] > 224 and imgArea[randx,randy][2] > 224):
					white_val += 1
	
	return white_val

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

	for x in range((W)):
		for y in range((H)):
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

	for x in range((W)):
		for y in range((H)):
			if 	coordVals[x][y] == 1: # wall and unwanted area
				# WALL
				wall += 1

	clear_area = (h * w) - wall

	qty = round(clear_area / (weightCCTV*(math.pi * math.pow((CCTV_RADIUS/scale),2))))
	
	print("\nInitializing cctv quantity: ", qty, "\n")
	return qty

def randomizer():
	randx = random.randint(10,W-1)
	randy = random.randint(10,H-1)
	rand = (randx,randy)
	return randx, randy, rand

def randCoords(index,max_cctv):

	imgOutline = cv2.imread(raw_path,1)

	# rand_val = [[0 for y in range(0,H)] for x in range(0,W)]

	rand_list = []

	# generate genes

	while len(rand_list) < max_cctv: # cctv quan

		randx, randy, rand = randomizer()

		if (randx < W and randy < H) and (coordVals[randx][randy] != 1) and rand not in rand_list:

			if coordVals[randx][randy] != 1:

				if rand not in rand_list:
					
					if 	(imgOutline[randx,randy][0] > 223 and imgOutline[randx,randy][1] > 223 and imgOutline[randx,randy][2] > 223):
						rand_list.append(rand)
						radius = int(CCTV_DIST_WEIGHT*CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgOutline = circleArea(imgOutline, rand, radius, (0, 191, 0), -1)
	
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
	

	# imgCoord = cv2.imread(raw_path)
	# for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		# imgCoord = circleCoord(imgCoord, rand, 2, (191, 0, 0), -1)
		
		# print(str(rand)+" : "+str(img[randx,randy][:]))
		# print(coord_vals[rand[0]][rand[1]])

	return rand_list

def chrValue(index,chromosomes): # colouring the image using value

	imgArea = cv2.imread(raw_path)
	# imgOutline = cv2.imread(raw_path)

	radius = int(CCTV_RADIUS/default_scale)
	gene_val = [[0 for y in range((H))] for x in range((W))]

	for gene in chromosomes:
		for x in range((W)):
			for y in range((H)):
				if 	coordVals[x][y] == 1:
					# walls
					imgArea[x,y] = (7, 0, 11)
			
				if 	coordVals[x][y] == 0:
					# not walls
					imgArea[x,y] = (255, 255, 255)

	for gene in chromosomes:
		# rand[1],rand[0] for image !!!!!!!!!!!

		imgArea = circleArea(imgArea, gene, radius, (0, 191, 0), -1)

		circle_x = gene[0]
		circle_y = gene[1]
		rad = int(CCTV_RADIUS/default_scale)

		for x in range((W)):
			for y in range((H)):

				if(isInside(circle_x, circle_y, rad, x, y)):
					# print("Inside")
					# area cover not wall
					if gene_val[x][y] == 0 and coordVals[x][y] != 1:
						gene_val[x][y] = 1

					# area cover wall
					if gene_val[x][y] == 0 and coordVals[x][y] == 1:
						imgArea[x,y] = (0,0,127) # red 

					# area overlap other area
					if gene_val[x][y] > 0:
						gene_val[x][y] += 1

					if gene_val[x][y] > 2:
						imgArea[x,y] = (127,127,0) # cyan

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----
	
	image_path = ('result/imgArea') + str(index) + '.png'
	cv2.imwrite(image_path,imgArea)

	image_path = ('result/imgValue') + str(index) + '.png'
	cv2.imwrite(image_path,imgArea)

def cal_pop_fitness(index,chr): # value[], randList[] calculate fitness using colour on image

	# find total white pixel

	available_path = 'result/availableArea.png'
	imgAvailable = cv2.imread(available_path,1)

	totalAvailableArea = 0

	for x in range((W)):
		for y in range((H)):
			if (imgAvailable[x,y][0] > 224 and imgAvailable[x,y][1] > 224 and imgAvailable[x,y][2] > 224):
			# if coordVals[x][y] == 0:
				totalAvailableArea += 1

	fitness = 0.0
	gene_val = [[0 for y in range((H))] for x in range((W))]
	gene_fitness = []

	worstFitness_gene = 1.0
	# worst_gene_idx = 0
	g = 0

	chrValue(index,chr)

	value_path = 'result/imgValue' + str(index) + '.png'
	imgValue = cv2.imread(value_path,1)
	imgOutline = cv2.imread(raw_path,1)

	# colour the image

	for gene in chr: # gene fitness

		radius = int(CCTV_RADIUS/default_scale)

		imgOutline = circleArea(imgOutline, gene, radius, (0, 128, 0), 1)
		imgOutline = circleCoord(imgOutline, gene, 2, (191, 0, 0), -1)

		circle_x = gene[0]
		circle_y = gene[1]
		rad = int(CCTV_RADIUS/default_scale)
		total_gene_val = 0
		gene_penalty = 0

		total_green = 0
		total_cyan = 0
		total_red = 0
		circle_area = 0

		for x in range((W)):
			for y in range((H)):

				if(isInside(circle_x, circle_y, rad, x, y)):

					circle_area += 1

					if (imgValue[x,y][0] < 64 and imgValue[x,y][1] > 128 and imgValue[x,y][2] < 64):	# green 0 191 0
						total_green += 1

					if (imgValue[x,y][0] > 64 and imgValue[x,y][1] > 64 and imgValue[x,y][2] < 64):		# cyan 127 127 0
						total_cyan += OVERLAP_PENALTY

					if (imgValue[x,y][0] < 64 and imgValue[x,y][1] < 64 and imgValue[x,y][2] > 64):		# red 0 0 127
						total_red += WALL_PENALTY

		# total_gene_val = (total_green/circle_area) - (total_cyan + total_red) / circle_area
		# total_gene_val = (total_green + total_cyan) / circle_area
		total_gene_val = (total_green) / circle_area # smaller green, smaller fitness
		gene_fitness.append(float(total_gene_val))

		if total_gene_val < worstFitness_gene:
			worstFitness_gene = total_gene_val
			worst_gene = gene
		
		# cv2.putText(imgArea, str(i), (gene[1],gene[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.putText(imgOutline, str(gene), (gene[1],gene[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)

		g+=1

	# chromosome fitness
	for x in range((W)):
		for y in range((H)):

			if (imgValue[x,y][0] < 64 and imgValue[x,y][1] > 128 and imgValue[x,y][2] < 64):	# green 0 191 0
				total_green += 1

			if (imgValue[x,y][0] > 64 and imgValue[x,y][1] > 64 and imgValue[x,y][2] < 64):		# cyan 127 127 0
				total_cyan += OVERLAP_PENALTY

			if (imgValue[x,y][0] < 64 and imgValue[x,y][1] < 64 and imgValue[x,y][2] > 64):		# red 0 0 127
				total_red += WALL_PENALTY

	# fitness = (total_green/totalAvailableArea) - (total_cyan + total_red)/totalAvailableArea
	fitness = (total_green + total_cyan) / totalAvailableArea # coverage percentage

	print(str(fitness),str(total_green),str(total_cyan),str(total_red),str(totalAvailableArea))

	cv2.imwrite(('result/imgValue' + str(index) + '.png'),imgValue)
	cv2.imwrite(('result/imgOutline' + str(index) + '.png'),imgOutline)

	# ----- ----- ----- ----- -----

	img2 = cv2.imread('result/bw.png',1)
	dst = cv2.addWeighted(img2, 0.6, imgValue, 0.6,0)
	
	cv2.imwrite('result/imgOverlay'+str(index)+'.png',dst) # double exposure image

	# ----- ----- ----- ----- -----
    
	return fitness, worstFitness_gene, worst_gene
	# return fitnessg
    # Calculating the fitness value of each solution in the current population.

def selection(population, fitness_values):
    # Perform selection based on worst gene fitness value each solution
    # to create a new population for the next generation
    # use roulette wheel selection
	li=[]
	parents=[]
 
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

	# indices = sort_index[:4]

	print(indices)

	for i in range(4):
		parents.append(population[indices[i]])
		print(parents[i])

	return parents, indices

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

def mutation(parent, parent_idx, worst_gene):

	# for i in worst_sol_idx:
	# 	print(i)

	mutated_offspring = []
	offspring = []

	# create offspring using parent idx

	offspring1 = parent[0]
	offspring2 = parent[1]
	offspring3 = parent[2]
	offspring4 = parent[3]

	# worst gene idx in each offspring

	gene1 = worst_gene[parent_idx[0]]
	gene2 = worst_gene[parent_idx[1]]
	gene3 = worst_gene[parent_idx[2]]
	gene4 = worst_gene[parent_idx[3]]
	
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

	for i in parent:
		print(i)
		offspring.append(i)

	# for i in parent_idx:
	# 	print(i)
	# for i in worst_gene:
	# 	print(i)

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	old_white_val1 = 0
	old_white_val2 = 0
	old_white_val3 = 0
	old_white_val4 = 0

	# ---

	# for gene in offspring1:
	# 	imgArea1 = circleArea(imgArea1, gene, radius, (0, 191, 0), -1)

	# max_white = 0
	# white_val1 = -1

	# while white_val1 < max_white:
	# 	randx, randy, rand = randomizer()
	# 	white_val1 = calculateWhite(randx, randy, imgArea1, randx, randy)

	# 	if (randx < W and randy < H) and coordVals[randx][randy] != 1 and rand not in offspring1 and white_val1 > max_white:
	# 		max_white = white_val1

	# new_gene = rand

	# print(white_val1, max_white, rand)
	# imgArea1 = circleArea(imgArea1, (randy,randx), radius, (0, 191, 0), 1)

	# cv2.imshow('area1', imgArea1)
	# cv2.waitKey(0)

	#  ---


	for gene in offspring1:
		if gene != gene1:
			imgArea1 = circleArea(imgArea1, gene, radius, (0, 191, 0), -1)
			count1 += 1
			
		else:
			circle_x = gene1[0]
			circle_y = gene1[1]
			rad = int(CCTV_RADIUS/default_scale)

			for x in range((W)):
				for y in range((H)):

					if(isInside(circle_x, circle_y, rad, x, y)):
						if (imgArea1[x,y][0] > 224 and imgArea1[x,y][1] > 224 and imgArea1[x,y][2] > 224):
							old_white_val1 += 1
			

	for gene in offspring2:
		if gene != gene2:
			imgArea2 = circleArea(imgArea2, gene, radius, (0, 191, 0), -1)
			count2 += 1
		else:
			circle_x = gene2[0]
			circle_y = gene2[1]
			rad = int(CCTV_RADIUS/default_scale)

			for x in range((W)):
				for y in range((H)):

					if(isInside(circle_x, circle_y, rad, x, y)):
						if (imgArea2[x,y][0] > 224 and imgArea2[x,y][1] > 224 and imgArea2[x,y][2] > 224):
							old_white_val2 += 1

	for gene in offspring3:
		if gene != gene3:
			imgArea3 = circleArea(imgArea3, gene, radius, (0, 191, 0), -1)
			count3 += 1
		else:
			circle_x = gene3[0]
			circle_y = gene3[1]
			rad = int(CCTV_RADIUS/default_scale)

			for x in range((W)):
				for y in range((H)):

					if(isInside(circle_x, circle_y, rad, x, y)):
						if (imgArea3[x,y][0] > 224 and imgArea3[x,y][1] > 224 and imgArea3[x,y][2] > 224):
							old_white_val3 += 1

	for gene in offspring4:
		if gene != gene4:
			imgArea4 = circleArea(imgArea4, gene, radius, (0, 191, 0), -1)
			count4 += 1
		else:
			circle_x = gene4[0]
			circle_y = gene4[1]
			rad = int(CCTV_RADIUS/default_scale)

			for x in range((W)):
				for y in range((H)):	

					if(isInside(circle_x, circle_y, rad, x, y)):
						if (imgArea4[x,y][0] > 224 and imgArea4[x,y][1] > 224 and imgArea4[x,y][2] > 224):
							old_white_val4 += 1

	path1 = 'result/imgMutation0.png'
	path2 = 'result/imgMutation1.png'
	path3 = 'result/imgMutation2.png'
	path4 = 'result/imgMutation3.png'

	cv2.imwrite(path1,imgArea1)
	cv2.imwrite(path2,imgArea2)
	cv2.imwrite(path3,imgArea3)
	cv2.imwrite(path4,imgArea4)

	# print(count1)

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	# cv2.imshow('imgArea1',imgArea1)
	
	# k = cv2.waitKey(0)
	# if k == ord("s") or k == ord("S"):
	# 	cv2.destroyAllWindows

	radius = int(CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
	areaCircle = 22.0/7.0 * radius ** 2

	while len(offspring1) > count1:

		circle_x = gene1[0]
		circle_y = gene1[1]

		randx, randy, rand = randomizer()

		white_val1 = calculateWhite(circle_x, circle_y, path1, randx, randy)

		if ((randx < W and randy < H) and coordVals[randx][randy] != 1 and rand not in offspring1 
      		and isInside(circle_x, circle_y, int(2.0*CCTV_RADIUS/default_scale), randx, randy) 
	  		and (imgArea1[randx,randy][0] > 224 and imgArea1[randx,randy][1] > 224 and imgArea1[randx,randy][2] > 224) 
	  		and (white_val1 >= (old_white_val1*WHITE_RANGE_WEIGHT) and white_val1 > 0.333*areaCircle)):

			print(white_val1, old_white_val1, areaCircle)

			for i in range(len(offspring1)):
				if offspring1[i] == gene1:
					offspring1[i] = rand
					print(str(offspring1[i]) + " mutated")

					print(str(gene1)+" >>> "+str(rand) + "\t Solution " + str(parent_idx[0]) + "\n")
					imgArea1 = circleArea(imgArea1, rand, radius, (0, 191, 0), -1)
					count1 += 1

	while len(offspring2) > count2:

		circle_x = gene2[0]
		circle_y = gene2[1]

		randx, randy, rand = randomizer()

		white_val2 = calculateWhite(circle_x, circle_y, path2, randx, randy)

		if ((randx < W and randy < H) and coordVals[randx][randy] != 1 and rand not in offspring2 
      		and isInside(circle_x, circle_y, int(2.0*CCTV_RADIUS/default_scale), randx, randy) 
	  		and (imgArea2[randx,randy][0] > 224 and imgArea2[randx,randy][1] > 224 and imgArea2[randx,randy][2] > 224) 
	  		and (white_val2 >= (old_white_val2*WHITE_RANGE_WEIGHT) and white_val2 > 0.333*areaCircle)):

			print(white_val2, old_white_val2, areaCircle)

			for i in range(len(offspring2)):
				if offspring2[i] == gene2:
					offspring2[i] = rand
					print(str(offspring2[i]) + " mutated")

					print(str(gene2)+" >>> "+str(rand) + "\t Solution " + str(parent_idx[1]) + "\n")
					imgArea2 = circleArea(imgArea2, rand, radius, (0, 191, 0), -1)
					count2 += 1

	while len(offspring3) > count3:

		circle_x = gene3[0]
		circle_y = gene3[1]

		randx, randy, rand = randomizer()

		white_val3 = calculateWhite(circle_x, circle_y, path3, randx, randy)

		if ((randx < W and randy < H) and coordVals[randx][randy] != 1 and rand not in offspring3 
      		and isInside(circle_x, circle_y, int(2.0*CCTV_RADIUS/default_scale), randx, randy) 
	  		and (imgArea3[randx,randy][0] > 224 and imgArea3[randx,randy][1] > 224 and imgArea3[randx,randy][2] > 224) 
	  		and (white_val3 >= (old_white_val3*WHITE_RANGE_WEIGHT) and white_val3 > 0.333*areaCircle)):

			print(white_val3, old_white_val3, areaCircle)

			for i in range(len(offspring3)):
				if offspring3[i] == gene3:
					offspring3[i] = rand
					print(str(offspring3[i]) + " mutated")

					print(str(gene3)+" >>> "+str(rand) + "\t Solution " + str(parent_idx[2]) + "\n")
					imgArea3 = circleArea(imgArea3, rand, radius, (0, 191, 0), -1)
					count3 += 1

	while len(offspring4) > count4:

		circle_x = gene4[0]
		circle_y = gene4[1]

		randx, randy, rand = randomizer()

		white_val4 = calculateWhite(circle_x, circle_y, path4, randx, randy)

		if ((randx < W and randy < H) and coordVals[randx][randy] != 1 and rand not in offspring4 
      		and isInside(circle_x, circle_y, int(2.0*CCTV_RADIUS/default_scale), randx, randy) 
	  		and (imgArea4[randx,randy][0] > 224 and imgArea4[randx,randy][1] > 224 and imgArea4[randx,randy][2] > 224) 
	  		and (white_val4 >= (old_white_val4*WHITE_RANGE_WEIGHT) and white_val4 > 0.333*areaCircle)):

			print(white_val4, old_white_val4, areaCircle)

			for i in range(len(offspring4)):
				if offspring4[i] == gene4:
					offspring4[i] = rand
					print(str(offspring4[i]) + " mutated")

					print(str(gene4)+" >>> "+str(rand) + "\t Solution " + str(parent_idx[3]) + "\n")
					imgArea4 = circleArea(imgArea4, rand, radius, (0, 191, 0), -1)
					count4 += 1

	# print(offspring1)
	# print(offspring2)

	cv2.imwrite(('result/imgArea' + str(parent_idx[0]) + '.png'),imgArea1)
	cv2.imwrite(('result/imgArea' + str(parent_idx[1]) + '.png'),imgArea2)
	cv2.imwrite(('result/imgArea' + str(parent_idx[2]) + '.png'),imgArea3)
	cv2.imwrite(('result/imgArea' + str(parent_idx[3]) + '.png'),imgArea4)

	chrValue(parent_idx[0],offspring1)
	chrValue(parent_idx[1],offspring2)
	chrValue(parent_idx[2],offspring3)
	chrValue(parent_idx[3],offspring4)

	mutated_offspring.append(offspring1)
	mutated_offspring.append(offspring2)
	mutated_offspring.append(offspring3)
	mutated_offspring.append(offspring4)

	return mutated_offspring

# def CCTV(chr_size):
# 	for x in range(sol_per_pop):
# 		chr = randCoords(x, chr_size)
# 		initial_population.append(chr)

# 		chrValue(x,chr)

# 	print("\n... Initial Population ...\n")
# 	for i in initial_population:
# 		print(i)

# 	# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# 	print("\n... Fitness Function ...\n")

# 	worst_sol_fitness = [] 	# worst gene fitness each sol
# 	worst_sol_idx = []		# worst gene idx each sol

# 	worst_gene_fitness = 0
# 	worst_gene_idx = 0

# 	best_chr_fitness = 0
	
# 		# # Measing the fitness of each chromosome in the solution.
# 	for x in range(sol_per_pop):
# 		chrFitness, worst_gene_fitness, worst_gene_idx = cal_pop_fitness(x,current_population[x])
			
# 		fitness.append(chrFitness)

# 			# selection, crossover, mutation variables for GENES
# 		worst_sol_fitness.append(worst_gene_fitness)
# 		worst_sol_idx.append(worst_gene_idx)

# 		print("\nfitness "+str(x)+" : "+str(chrFitness)+"\tworst gene "+str(worst_gene_idx)+" : "+str(current_population[x][worst_gene_idx])+"\n")
# 			# print("worst fitness "+str(x)+" : "+str(worst_gene_fitness)+"\n")

# 		if chrFitness > best_chr_fitness:
# 			best_chr_fitness = chrFitness
# 			best_chr_idx = x
# 			best_chr = current_population[best_chr_idx]

# 		chrValue(x,current_population[x])

# 	average_fitness = sum(fitness) / len(fitness)
		
# 	print("\nBest result : ",best_chr)
# 	print("Best fitness "+str(best_chr_idx)+" : ",best_chr_fitness)
# 	print("Average fitness : ",average_fitness,"\n")

# 	return 0

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

	availableArea_image = original_image

	# availableArea_image = selectROI_area(original_image)

	# stop = False
	# while stop == False:
	# 	print("\nPress 's' to stop")

	# 	k = cv2.waitKey(0)
	# 	if k == ord("s") or k == ord("S"):
	# 		stop = True
	# 		print("\nFinalizing available area ...")
	# 	else:
	# 		availableArea_image = selectROI_area(availableArea_image)

	# cv2.destroyAllWindows()

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	# PARAMETER SETTINGS
	convergence_threshold = 0.02
	frequency = 0

	# assign value to coord
	# 1 = wall, remove
	# 0 = empty
	# validation only
	# Implement area_valuer() to assign values in a 2D array 
	# representing the coordinates in the resulting image from the selectROI() loop.

	areaValuer() # coordVals 

	# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

	# to initialize the CCTV quantity (solution) based on the available values and estimated total CCTV coverage.
	chr_size = initializeSol(default_scale, weightCCTV)
	
	initial_population = []
	fitness = []
	
	sol_per_pop = 8

	previous_fitness = 0.000
	best_sol_fitness = -999
	best_sol_idx = 0
	best_sol = []

	# Measing the fitness of each chromosome in the population.
	for x in range(sol_per_pop):
		chr = randCoords(x, chr_size)
		initial_population.append(chr)

		chrValue(x,chr)

	print("\n... Initial Population ...\n")
	for i in initial_population:
		print(i)

	current_population = initial_population
	decreasing_fitness_count = 0
	up_cctv_threshold = 5 # cctv inc after 5 times dec fitness
	
	for generation in range(NUM_GENERATION):
		print("\n... Generation : ", generation+1," ...\n")

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		print("\n... Fitness Function ...\n")

		worst_sol_fitness = [] 	# worst gene fitness each sol
		worst_sol_gene = []		# worst gene idx each sol

		# worstFitness_sol = 0
		# worst_gene_idx = 0

		best_chr_fitness = 0
	
		# # Measing the fitness of each chromosome in the solution.
		for x in range(sol_per_pop):
			chrFitness, worstFitness_sol, worst_gene = cal_pop_fitness(x,current_population[x])
			
			fitness.append(chrFitness)

			# selection, crossover, mutation variables for GENES
			worst_sol_fitness.append(worstFitness_sol)
			worst_sol_gene.append(worst_gene)

			print("\nfitness "+str(x)+" : "+str(chrFitness)+"\tworst gene "+str(worst_gene)+"\n")
			# print("worst fitness "+str(x)+" : "+str(worst_gene_fitness)+"\n")

			if chrFitness > best_chr_fitness:
				best_chr_fitness = chrFitness
				best_chr_idx = x
				best_chr = current_population[best_chr_idx]

			# chrValue(x,chr)

		average_fitness = sum(fitness) / len(fitness)
		
		print("\nBest result : ",best_chr)
		print("Best fitness "+str(best_chr_idx)+" : ",best_chr_fitness)
		print("Average fitness : ",average_fitness,"\n")

		# save best sol every gen
		best_gen_sol = 'result/imgOutline' + str(best_chr_idx) + '.png'
		imgBestGenSol = cv2.imread(best_gen_sol,1)
		# best_gen_sol_path = 'result/imgBestSolGen' + str(generation) + str(best_chr_idx) + '.png'
		best_gen_sol_path = 'result/test_result.png'
		cv2.imwrite(best_gen_sol_path, imgBestGenSol)
		# cv2.imshow('result',bestSolImg)

		# update result, fitness, average

		# retrieve previous gen best fitness
		if os.path.exists("FITTEST.txt"):
			fname = 'FITTEST.txt'
			previous_fitness = float(LastNlines(fname))

		# FIND THE FITTEST
		if best_chr_fitness > best_sol_fitness:
			best_sol_fitness = best_chr_fitness
			best_sol_idx = best_chr_idx
			best_sol = best_chr

			# update FITTEST img
			best_image_path = 'result/imgBestSolution.png'
			cv2.imwrite(best_image_path,imgBestGenSol)

			frequency = 1
		
		if best_sol_fitness == previous_fitness:
			frequency += 1
			print("\nFittest freq : " + str(frequency) + "\n")

		# if best_chr_fitness < previous_fitness:
		# 	if decreasing_fitness_count < up_cctv_threshold:
		# 		decreasing_fitness_count += 1
		# 	# else:
			# 	chr_size += 1
			# 	initial_population = CCTV(chr_size)


		# write result, fitness, average to text file PER GENERATION
		bestResult = open("RESULT.txt", "a")
		bestResult.write(str(best_chr)+"\n")
		bestResult.close()

		bestFitness = open("FITNESS.txt", "a")
		bestFitness.write(str(best_chr_fitness)+"\n")
		bestFitness.close()

		bestFitness = open("FITTEST.txt", "a")
		bestFitness.write(str(best_sol_fitness)+"\n")
		bestFitness.close()

		avgFitness = open("AVERAGE.txt", "a")
		avgFitness.write(str(average_fitness)+"\n")
		avgFitness.close()

		print("\n... Text file updated ...\n")

		# reset 
		# best_sol_fitness.clear()
		# best_sol_idx.clear()
		# best_sol.clear()

		# print("\nworst gene fitness : ",worst_sol_fitness)
		# print("worst gene idx : ",worst_sol_idx,"\n")

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# TERMINATION CRITERION

		convergence = abs(best_sol_fitness - average_fitness)

		print("\nConvergence : " + str(convergence) + "\n")

		if (convergence > 0 and convergence < convergence_threshold) or frequency > 10:

			if (convergence > 0 and convergence < convergence_threshold):
				print("... Convergence Threshold Reached "+ str(convergence_threshold) +" ...")

			if frequency > 10:
				print("... Fittest Frequency Reached 10 Times ...")

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
			# cv2.imshow('result',imgBestSol)

			# k = cv2.waitKey(0)
			# if k == ord("s") or k == ord("S"):
			# 	cv2.destroyAllWindows
			print("\n... System Terminated ...\n\n")

			# After optimization is done, run dashboard.py
			subprocess.Popen(["python", "dashboard.py"])

			exit()

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		print("\n... Parent Selection ...\n")
		
		# Selecting the best parents in the population for mating. parents = (sol idx in pop)
		parents, indices = selection(current_population, fitness)
		for i in range(len(parents)):
			print("\nParent "+ str(indices[i]) +" : "+str(parents[i]))

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# Generating next generation using crossover.
		# No crossover. Copy parents to offsprings
		# offsprings = parents

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		print("\n... Offspring Mutation ...\n")

		# Adding some variations to the offsrping using mutation.
		mutated_offsprings = mutation(parents, indices, worst_sol_gene)

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# Creating the new population based on the parents and offspring.

		current_population[indices[0]] = mutated_offsprings[0]
		current_population[indices[1]] = mutated_offsprings[1]
		current_population[indices[2]] = mutated_offsprings[2]
		current_population[indices[3]] = mutated_offsprings[3]

		print("\n... Survival Population ...\n")
		for i in current_population:
			print(i)

		fitness.clear()
		worst_sol_fitness.clear()
		worst_sol_gene.clear()

		# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

		# generation end

# ---------------------------------------------------------

# Getting the best solution after iterating finishing all generations.
# print("\nBest solution : ",best_sol)
# print("Fitness : ",best_sol_fitness)
# radius = int(CCTV_RADIUS/default_scale)
# i=1
# imgBestSol = cv2.imread(raw_path,1)
# for gene in best_sol:
# 	imgBestSol = circleArea(imgBestSol, gene, radius, (0, 128, 0), 1)
# 	imgBestSol = circleCoord(imgBestSol, gene, 2, (191, 0, 0), -1)
# 	cv2.putText(imgBestSol, str(gene[1])+","+str(gene[0]), (gene[1],gene[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
# 	i += 1

# best_image_path = 'result/imgBestSolution.png'
# cv2.imwrite(best_image_path,imgBestSol)
# # best_image = cv2.imread(best_image_path,1)
# cv2.imshow('result',imgBestSol)

# k = cv2.waitKey(0)
# if k == ord("s") or k == ord("S"):
# 	cv2.destroyAllWindows
# print("\n... System Terminated ...\n\n")
# exit()








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
Implement the functions: selection, crossover, and mutation for the initial population in the genetic algorithm.
Implement a decision-making function to evaluate the best population's fitness value and determine whether to decrease, increase, or keep the initial CCTV quantity.
If the CCTV quantity needs to be changed, run the genetic algorithm again to find the best population with the new generated CCTV quantity.
Use a feed-forward or feed-backward technique in the decision-making process.


"""