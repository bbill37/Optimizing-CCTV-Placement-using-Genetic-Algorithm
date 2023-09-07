# Python program to demonstrate
# writing to CSV


import csv
# code for displaying multiple images in one figure

#import libraries
import cv2
from matplotlib import pyplot as plt

# create figure
fig = plt.figure(figsize=(0.8*16, 0.8*9))

# setting values to rows and column variables
r = 1
c = 2

# from matplotlib import pyplot as plt
from matplotlib import image as mpimg

fig.add_subplot(r, c, 2)
 
plt.title("Best Solution")
plt.xlabel("X pixel coordinate")
plt.ylabel("Y pixel coordinate")
 
image = mpimg.imread("result/imgBestSolution.png")
plt.imshow(image)
# plt.show()
	
# # reading images
# Image1 = cv2.imread('result/imgBestSolution.png')
# # Adds a subplot at the 1st position
# fig.add_subplot(r, c, 2)

# # showing image
# plt.imshow(Image1)
# plt.axis('off')
# plt.title("Best Solution")

bestFitness = []
maxFitness = []
avgFitness = []

fit = []
max = []
avg = []

# ----- -----

with open('FITNESS.txt') as file:
        for line in file:
            fit.append(float(line))
	    
for i in range(len(fit)):
	bestFitness.append([str(i), str(fit[i])])
	
# writing to csv file
with open("fitness.csv", 'w', newline='') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)
		
	# writing the data rows
	csvwriter.writerows(bestFitness)

csvfile.close()

# ----- -----

with open('FITTEST.txt') as file:
        for line in file:
            max.append(float(line))
	    
for i in range(len(max)):
	maxFitness.append([str(i), str(max[i])])
	
# writing to csv file
with open("fittest.csv", 'w', newline='') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)
		
	# writing the data rows
	csvwriter.writerows(maxFitness)

csvfile.close()

# ----- -----

with open('AVERAGE.txt') as file:
        for line in file:
            avg.append(float(line))
	    
for i in range(len(avg)):
	avgFitness.append([str(i), str(avg[i])])
	
# writing to csv file
with open("average.csv", 'w', newline='') as csvfile:
	# creating a csv writer object
	csvwriter = csv.writer(csvfile)
		
	# writing the data rows
	csvwriter.writerows(avgFitness)

csvfile.close()

# ----- -----

xf = []
yf = []
xm = []
ym = []
xa = []
ya = []

fig.add_subplot(r, c, 1)

# ----- -----

with open('fitness.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	for row in lines:
		xf.append(row[0])
		yf.append(float(row[1]))
		
plt.plot(xf, yf, color = 'r', linestyle = 'dashed',
		marker = 'o',label = "Best Fitness")

with open('fittest.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	for row in lines:
		xm.append(row[0])
		ym.append(float(row[1]))
	maxFitness = float(row[1])
		
plt.plot(xm, ym, color = 'g', linestyle = 'dashed',
		marker = 'o',label = "Fittest")

with open('average.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	for row in lines:
		xa.append(row[0])
		ya.append(float(row[1]))
		
plt.plot(xa, ya, color = 'y', linestyle = 'dashed',
		marker = 'o',label = "Average Fitness")

# ----- -----

plt.xticks(rotation = 25)
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title('GA Performance, max = '+str(maxFitness), fontsize = 20)
plt.grid()
plt.legend()
plt.show()

# fig.add_subplot(r, c, 3)

# plt.text(1, 0.05, 'CCTV_DIST_WEIGHT\nweightCCTV\nNUM_GENERATION\nWALL_PENALTY\nOVERLAP_PENALTY\nWHITE_RANGE_WEIGHT\n', 
# 	 fontsize = 10)

# plt.show()


# Image2 = cv2.imread('Image2.jpg')
# Image3 = cv2.imread('Image3.jpg')
# Image4 = cv2.imread('Image4.jpg')

# Adds a subplot at the 1st position
# fig.add_subplot(r, c, 2)

# showing image
# plt.imshow(Image1)
# plt.axis('off')
# plt.title("First")

# # Adds a subplot at the 2nd position
# fig.add_subplot(rows, columns, 2)

# # showing image
# plt.imshow(Image2)
# plt.axis('off')
# plt.title("Second")

# # Adds a subplot at the 3rd position
# fig.add_subplot(rows, columns, 3)

# # showing image
# plt.imshow(Image3)
# plt.axis('off')
# plt.title("Third")

# # Adds a subplot at the 4th position
# fig.add_subplot(rows, columns, 4)

# # showing image
# plt.imshow(Image4)
# plt.axis('off')
# plt.title("Fourth")
