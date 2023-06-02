import numpy as np
import pandas as pd
import cv2

def value(img,coord_vals):
	
	# for y in range(0,h):
	for x in range(0,w):
		# for x in range(0,w):
		for y in range(0,h):
			if 	img[x,y][2] < 128:
				coord_vals[x][y] = 1
			else:
				coord_vals[x][y] = 0

	if pd.DataFrame(coord_vals).to_csv('coord_vals.csv')==True: 
		print("coord_vals.csv saved")


raw_image = cv2.imread('a.png')
w, h, _ = raw_image.shape

# coord_vals = [[0 for x in range(0,w)] for y in range(0,h)]
coord_vals = [[0 for y in range(0,h)] for x in range(0,w)]

value(raw_image,coord_vals)