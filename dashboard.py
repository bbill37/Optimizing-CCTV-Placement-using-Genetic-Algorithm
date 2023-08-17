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

# reading images
Image1 = cv2.imread('result/imgBestSolution.png')
# Adds a subplot at the 1st position
fig.add_subplot(r, c, 1)

# showing image
plt.imshow(Image1)
plt.axis('off')
plt.title("Best Solution")
	
bestFitness = []
avgFitness = []

fit = []
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

import matplotlib.pyplot as plt
import csv

xf = []
yf = []
xa = []
ya = []

fig.add_subplot(r, c, 2)

with open('fitness.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	for row in lines:
		xf.append(row[0])
		yf.append(float(row[1]))
		
plt.plot(xf, yf, color = 'g', linestyle = 'dashed',
		marker = 'o',label = "Best Fitness")

with open('average.csv','r') as csvfile:
	lines = csv.reader(csvfile, delimiter=',')
	for row in lines:
		xa.append(row[0])
		ya.append(float(row[1]))
		
plt.plot(xa, ya, color = 'y', linestyle = 'dashed',
		marker = 'o',label = "Average Fitness")

plt.xticks(rotation = 25)
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title('GA Performance (10 Generations)', fontsize = 20)
plt.grid()
plt.legend()
plt.show()


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
