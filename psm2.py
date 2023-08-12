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
import os

RAW_PATH = "art.png"

# read floor plan image
def read_image():
	img = cv2.imread(RAW_PATH, 1)
	return img

def selectROI_area(image, output_filename='availableArea.png', mark_color=(0, 0, 64)):
    r = cv2.selectROI("select the area", image)
    image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[0]+r[2]), int(r[1]+r[3])), mark_color, -1)

    if cv2.imwrite(output_filename, image):
        print(f"\n{output_filename} updated")

    return image

# initialize cctv number based on area? -------------------------
def initialize_max_cctv(scale):
	# Load availableArea.png
    available_area_image = cv2.imread('availableArea.png', 1)  # Load as color image

    # Convert the available area image to grayscale
    gray_available_area = cv2.cvtColor(available_area_image, cv2.COLOR_BGR2GRAY)

    # Calculate the total available area
    total_available_pixels = cv2.countNonZero(gray_available_area)

    # Determine the area of a single CCTV coverage circle
    cctv_radius = 100  # Adjust this based on your problem
    single_cctv_area = 3.14159 * (cctv_radius ** 2)  # Area of a circle: πr^2

    # Calculate the maximum number of non-overlapping CCTV cameras
    max_cctv = int(total_available_pixels / single_cctv_area)

    print("\nMaximum CCTV quantity: ", max_cctv, "\n")
    return max_cctv

# ---------- ---------- ---------- ---------- ----------

def initialize_population(pop_size, max_cctv, W, H):
    population = []
    for _ in range(pop_size):
        individual = []
        while len(individual) < max_cctv:
            randx = random.randint(50, W)
            randy = random.randint(50, H)
            gene = (randx, randy)
            if is_valid_gene(gene, individual):
                individual.append(gene)
        population.append(individual)
    return population

def is_valid_gene(gene, individual):
    # Check if gene is within the available area (not in walls)
    # Check if gene is not a duplicate in the individual
    # Add any other constraints you want to impose
    return True

def fitness_function(individual):
    # Calculate the fitness value for the individual
    # based on the objectives and constraints of your problem
    return fitness_value

def selection(population, fitness_values):
    # Perform selection based on fitness values
    # to create a new population for the next generation
    # You can use different selection methods like tournament selection or roulette wheel selection
    return new_population

def crossover(parent1, parent2):
    # Perform crossover (recombination) to create offspring
    # based on parents' genes
    # You can use different crossover techniques like one-point crossover or uniform crossover
    return offspring

def mutation(individual, mutation_rate):
    # Perform mutation on individual's genes with the given mutation rate
    # You can apply random changes to genes to introduce diversity
    return mutated_individual

def genetic_algorithm(pop_size, max_cctv, W, H, generations, mutation_rate):
    # Initialization
    population = initialize_population(pop_size, max_cctv, W, H)
    print(population)
    
    for generation in range(generations):
        # Evaluate fitness for each individual
        fitness_values = [fitness_function(individual) for individual in population]

        # Selection
        population = selection(population, fitness_values)

        # Crossover
        new_population = []
        while len(new_population) < pop_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            offspring = crossover(parent1, parent2)
            new_population.append(offspring)

        # Mutation
        for i in range(pop_size):
            if random.random() < mutation_rate:
                new_population[i] = mutation(new_population[i], mutation_rate)

        population = new_population

    # Return the best individual found after all generations
    best_individual = max(population, key=lambda ind: fitness_function(ind))
    return best_individual

# ---------- ---------- ---------- ---------- ----------

# Main code
if __name__ == "__main__":
    # raw_path = "path/to/your/image.png"
    imgRaw = cv2.imread(RAW_PATH, 1)
    H, W, _ = imgRaw.shape

    imgAvailableArea = selectROI_area(imgRaw)
    
    stop = False
    while stop == False:
        print("\nPress 's' to stop")
        
        k = cv2.waitKey(0)
        if k == ord("s") or k == ord("S"):
            stop = True
            print("\nFinalizing available area ...")
        else:
            imgAvailableArea = selectROI_area(imgAvailableArea)

    cv2.destroyAllWindows()

    max_cctv = 8
    pop_size = 8
    generations = 1
    mutation_rate = 0.1

    best_individual = genetic_algorithm(pop_size, max_cctv, W, H, generations, mutation_rate)
    print("Best Individual:", best_individual)
