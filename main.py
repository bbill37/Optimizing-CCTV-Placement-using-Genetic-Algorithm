from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import cv2
import math
# from matplotlib import pyplot as plt
import pandas as pd
import sys
import random
import numpy as np
from array import *
import operator as op

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

#function which returns the index of minimum value in the list
def get_minidx(inputlist):
 
    #get the minimum value in the list
    min_value = min(inputlist)
 
    #return the index of minimum value 
    min_index=inputlist.index(min_value)
    return min_index
# ------------------------------------------------------------

# declaration
raw_path = "art.png"
coords = []
value = []
default_scale = 1

img = cv2.imread(raw_path, 1)
w, h, c = img.shape
W = w-1
H = h-1

coordVals = [[0 for y in range(0,H)] for x in range(0,W)]

CCTV_RADIUS = 1000
MIN_DISTANCE = 900
MAX_DISTANCE = 1100

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
def draw_rectangle(image,coords): # NOT USING
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

def click_event(event, x, y, flags, params): # NOT USING
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
def area_remover(img): # NOT USING
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

	availableImage = cv2.imread('availableArea.png',1)

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

	valuedImage = cv2.imwrite('bw.png',availableImage)

	if pd.DataFrame(coordVals).to_csv('coordVals.csv')==True: 
		print("coordVals.csv saved")

	return valuedImage

def selectROI_area(image):
	r = cv2.selectROI("select the area", image)

	image = cv2.rectangle(image, (int(r[0]),int(r[1])), 
		(int(r[0]+r[2]),int(r[1]+r[3])), (0,0,64), -1)

	if cv2.imwrite('availableArea.png', image) == True:
		print("\navailableArea.png updated")

	return image

# initialize cctv number based on area? -------------------------
def initializeSol(scale):
	wall = 0
	qty = 0

	for x in range(0,W):
		for y in range(0,H):
			if 	coordVals[x][y] == 1: # wall and unwanted area
				# WALL
				wall += 1

	clear_area = (h * w) - wall

	qty = round(clear_area / (0.8*(math.pi * math.pow((CCTV_RADIUS/scale),2))))
	
	print("\nInitializing cctv quantity: ", qty, "\n")
	return qty

def randCoords(index,max_cctv):

	imgRaw = cv2.imread(raw_path,1)

	# rand_val = [[0 for y in range(0,H)] for x in range(0,W)]

	rand_list = []

	randx = random.randint(50,W)
	randy = random.randint(50,H)
	rand = (randx,randy)

	while len(rand_list) < max_cctv: # cctv quan

		if randx < W and randy < H:

			if coordVals[randx][randy] != 1:

				if rand not in rand_list:
					
					if 	(imgRaw[randx,randy][0] < 192 and imgRaw[randx,randy][1] < 192 and imgRaw[randx,randy][2] < 192):
						randx = random.randint(50,W)
						randy = random.randint(50,H)
						rand = (randx,randy)
					
					else:
						rand_list.append(rand)
						radius = int(1.5*CCTV_RADIUS/default_scale) # DISTANCE BTWN GENES
						imgRaw = circleArea(imgRaw, rand, radius, (0, 191, 0), -1)
					
				else:
					randx = random.randint(0,W)
					randy = random.randint(0,H)
					rand = (randx,randy)

			else:
				randx = random.randint(0,W)
				randy = random.randint(0,H)
				rand = (randx,randy)
				
		else:
			randx = random.randint(0,W)
			randy = random.randint(0,H)
			rand = (randx,randy)
	
	print("Random Coordinates Generated ...\n")

	radius = int(CCTV_RADIUS/default_scale)

	imgArea = cv2.imread(raw_path)
	for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		imgArea = circleArea(imgArea, rand, radius, (0, 191, 0), -1)

	image_path = 'imgArea' + str(index) + '.png'
	cv2.imwrite(image_path,imgArea)
	# cv2.imwrite('imgArea.png',imgArea)

	imgRaw = cv2.imread(raw_path)
	i = 1
	for rand in rand_list:

		# rand[1],rand[0] for image !!!!!!!!!!!
		imgAreaOutline = circleArea(imgRaw, rand, radius, (0, 128, 0), 1)
		imgAreaOutline = circleCoord(imgRaw, rand, 2, (191, 0, 0), -1)
		cv2.putText(imgRaw, str(i), (rand[1],rand[0]), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), 1)
		i += 1

	cv2.imwrite(('imgAreaOutline' + str(index) + '.png'),imgAreaOutline)
	

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
	img2 = cv2.imread('bw.png',1)
	dst = cv2.addWeighted(img2, 0.6, imgArea, 0.6, 0)

	# img_arr = np.hstack((img, img2))
	# cv2.imshow('Input Images',img_arr)
	# cv2.imshow('Blended Image',dst)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#---------------------------------------------------
	
	cv2.imwrite('rand'+str(index)+'.png',dst) # double exposure image

	return rand_list

def chrValue(index,chrList):
	
	avlImg = cv2.imread('bw.png',1)
	radius = int(CCTV_RADIUS/default_scale)
	for chr in chrList:
		# rand[1],rand[0] for image !!!!!!!!!!!
		avlImg = circleArea(avlImg, chr, radius, (0, 191, 0), -1)
	
	image_path = ('imgArea') + str(index) + '.png'
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
				# walls
				img[x,y] = (255, 255, 255)

	image_path = ('imageValue') + str(index) + '.png'
	cv2.imwrite(image_path,img)

	raw_image = cv2.imread(raw_path)

	for x in range(0,W):
		for y in range(0,H):
			if 	coordVals[x][y] == 1:
				# walls
				raw_image[x,y] = (7, 0, 11)

			if 	coordVals[x][y] == 2:
				# area
				raw_image[x,y] = (0, 191, 0)
			
			if 	coordVals[x][y] == 0:
				# available
				raw_image[x,y] = (255, 255, 255)

	# cv2.imwrite('rawValue.png',raw_image)
	image_path = 'rawValue' + str(index) + '.png'
	# cv2.imwrite(image_path,raw_image)

# genetic algorithm ----------------------------------------
def cal_pop_fitness(index,chr): # value[], randList[]

	rand_path = 'rand'+str(index)+'.png'
	imgValue = cv2.imread(rand_path,1)

	fitness = 0
	covered_coordinates = 0
	coverage_percentage = 0.0
	unwanted_penalty = 0
	coverable = 0
	total_coordinates = 0
	gene_val = [[0 for y in range(0,H)] for x in range(0,W)]
	gene_fitness = []

	for gene in chr:
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
						gene_penalty = op.add(gene_penalty,0.3)

					# area overlap
					if gene_val[x][y] > 0:
						gene_val[x][y] += 1
						gene_penalty = op.add(gene_penalty,0.3)

					# if coordVals[x][y] == 1:
					# 	gene_penalty = op.add(gene_penalty,1)
					# else:
					# 	total_gene_val = op.add(total_gene_val,1)

		total_gene_val = total_gene_val - gene_penalty
		gene_fitness.append(int(total_gene_val))

	# print(f"gene_fitness: ",gene_fitness)

	# NORMAL METHOD
	for x in range(0,W):
		for y in range(0,H):
		
			# if dark green value 1, penalty wall
			if 	(imgValue[x,y][0] < 64 and imgValue[x,y][1] > 64 and imgValue[x,y][2] < 64):
				unwanted_penalty = op.add(unwanted_penalty,1)
				# print(1)

			# if light green value 0, value set 2 and covered
			if 	(imgValue[x,y][0] < 192 and imgValue[x,y][1] > 192 and imgValue[x,y][2] < 192):
			# if 	(imgValue[x,y][0] < 64 and imgValue[x,y][1] > 128 and imgValue[x,y][2] < 64):

				# coverage
				# coordVals[x][y] = 2
				covered_coordinates = op.add(covered_coordinates,1)
				total_coordinates = op.add(total_coordinates,1)
				# print(2)

			if coordVals[x][y] != 1:
				# coordVals[x][y] = 0
				# imgValue[x,y] = (255,255,255)
				total_coordinates = op.add(total_coordinates,1)
				coverable += 1
				# print(0)
				

	# method 1
	coverage_percentage = covered_coordinates / total_coordinates
	fitness = coverage_percentage - (unwanted_penalty/total_coordinates)
	
	# method 2
	fitnessg = sum(gene_fitness) / coverable

	# print(covered_coordinates)
	# print(total_coordinates)
	# print(unwanted_penalty)
	# print("Formula 1 fitness: ",fitness)
	# print("Formula 2 fitness: ",fitnessg)

	# print("\nfitness value calculated ...")
    
	return fitness
	# return fitnessg
    # Calculating the fitness value of each solution in the current population.
	
def select_mating_pool(pop, fitness, num_parents):
	# Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
	parents = [0]*num_parents
	# parents = np.array(pop)
	# max_value = 0

	# RANK SELECTION
	sort_index = np.argsort(fitness)
	# print(sort_index)

	sort_index = sort_index[::-1]
	# print(sort_index)

	for parent_num in range(num_parents):
		parents[parent_num] = pop[sort_index[parent_num]]

	# for parent_num in range(num_parents):
		# if fitness[parent_num] < max_value:
			
			# max_value = fitness[parent_num]
			# parents.append(pop[parent_num])

	# print(f'',fitness)
	# print(f'',parents)

	# min_index = []
	# min_index.clear()

	# for i in parents:
	# 	min_fitness = min(fitness[i])
	# 	min_index.append(parents.index(min_fitness))

	# print(min_index)

	return parents

def crossover(parents, offspring_size):
	# offspring = [0]*offspring_size
	offspring = []
	crossover_point = offspring_size-4 # index 6

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

def mutation(pop):

	# print(pop[0])
	# print(pop[1])

	for chr in pop:

		gene_val = [[0 for y in range(0,H)] for x in range(0,W)]
		gene_fitness = []
		for gene in chr:
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
							gene_penalty = op.add(gene_penalty,0.3)

						# area overlap
						if gene_val[x][y] > 0:
							gene_val[x][y] += 1
							gene_penalty = op.add(gene_penalty,0.3)

			total_gene_val = total_gene_val - gene_penalty
			gene_fitness.append(int(total_gene_val))

		# mutation, remove least value coord
		min_val = min(gene_fitness)
		min_fitness_idx = gene_fitness.index(min_val)

		# print(chr[min_fitness_idx]," : ",min_fitness_idx," : ",min_val)

		# print(gene_fitness)

		chr.pop(min_fitness_idx)

		x = random.randint(0,W)
		y = random.randint(0,H)
		rand = (x,y)

		while (gene_val[x][y] != 0):
			
			x = random.randint(0,W)
			y = random.randint(0,H)
			rand = (x,y)

		# mutation, replace with new coord
		chr.append(rand)

	# print(pop[0])
	# print(pop[1])

	return pop

# ------------------------------------------------------------------------------------------------------

# driver function
if __name__=="__main__":
	# scale = 10 , 1pixel:10cm  or 20pixel:6feet

	# Read floor plan image from the directory.
	original_image = read_image()
	raw_path = "art.png"

	default_scale = 10

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

	# assign value to coord
	# 1 = wall, remove
	# 0 = empty
	# validation only
	# Implement area_valuer() to assign values in a 2D array 
	# representing the coordinates in the resulting image from the selectROI() loop.

	valuedImage  = areaValuer()

	# print(len(value))
	# print(len(value[0]))

	# print(value[:2])

	# if cv2.imwrite('after.png', availableArea_image) == True:
	# 	print("\nafter.png saved")

	

	# to initialize the CCTV quantity (solution) based on the available values and estimated total CCTV coverage.
	chr_size = initializeSol(default_scale)

	

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
	new_population = []
	fitness = []
	
	sol_per_pop = 4
	num_parents_mating = 2
	
	num_generations = 10
	# for generation in range(num_generations):
	# 	print("Generation : ", generation)
	
	# Measing the fitness of each chromosome in the population.
	for x in range(sol_per_pop):
		# print("\n----------------------------------------------------------------")
		# print("\nChromosome " + str(x+1) + ":\n")
		chr = randCoords(x,chr_size)
		new_population.append(chr)

		chrFitness = cal_pop_fitness(x,new_population[x])
		fitness.append(chrFitness)

		chrValue(x,chr)
	
	
	# Selecting the best parents in the population for mating.
	parents = select_mating_pool(new_population, fitness, num_parents_mating)

	# Generating next generation using crossover.
	offspring_crossover = crossover(parents,chr_size)

	# Adding some variations to the offsrping using mutation.
	offspring_mutation = mutation(offspring_crossover)

	# Creating the new population based on the parents and offspring.
	new_population[0:1] = parents
	new_population[2:] = offspring_mutation

	# print(new_population)

	# The best result in the current iteration.
	for x in range(sol_per_pop):
		chrFitness = cal_pop_fitness(x,new_population[x])
		fitness.append(chrFitness)
		chrValue(x,chr)

	average_fitness = sum(fitness) / len(fitness)

	best_chr_fitness = max(fitness)
	best_chr_idx = fitness.index(best_chr_fitness)
	best_chr = new_population[best_chr_idx]
	print("Best result : ",best_chr)
	print("Best fitness : ",fitness[best_chr_idx])
	print("Average fitness : ",average_fitness)

	best_image_path = 'imgAreaOutline'+str(best_chr_idx)+'.png'
	best_image = cv2.imread(best_image_path,1)
	cv2.imshow('result',best_image)

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