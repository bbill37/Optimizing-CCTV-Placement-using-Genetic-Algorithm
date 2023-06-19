import numpy
import ga

# This project is extended and a library called PyGAD is released to build the genetic algorithm.
# PyGAD documentation: https://pygad.readthedocs.io
# Install PyGAD: pip install pygad
# PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython

def cal_pop_fitness(equation_inputs, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    # print("aaa")
    # print(fitness)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # print(f"pop: ",pop)
    # print(f"pop: ",pop.shape[0])
    # print(f"pop: ",pop.shape[1])

    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    # print(f"parents: ",parents)
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        # print(f"max_fitness_idx: ",max_fitness_idx)
        max_fitness_idx = max_fitness_idx[0][0]
        # print(f"max_fitness_idx: ",max_fitness_idx)
        parents[parent_num, :] = pop[max_fitness_idx, :]
        print(parents[parent_num, :])
        fitness[max_fitness_idx] = -99999999999

    # print(f"pop: ",parents)
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)

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
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 4] = offspring_crossover[idx, 4] + random_value
    return offspring_crossover

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""

# Inputs of the equation.
equation_inputs = [4,-2,3.5,5,-11,-4.7]

# Number of the weights we are looking to optimize.
num_weights = 6

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 8
num_parents_mating = 4

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population)

num_generations = 5
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measing the fitness of each chromosome in the population.
    fitness = ga.cal_pop_fitness(equation_inputs, new_population)

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, 
                                      num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # The best result in the current iteration.
    print("Best result : ", numpy.max(numpy.sum(new_population*equation_inputs, axis=1)))

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


# -----------------------------------
coord = (10,10)
coord = ((coord[0]+1),(coord[1]+1))
print(coord)

# ---------------
s = [1,2,3,4,5,6,7,8,9]
p = [11,22,33,44,55,66,77,88,99]
s[:2] = p[-2:]
s[6:] = p[:3]
print(s)
# print(d)
crossover_point=5

a = []
a = s[0:4]
b = []
b = p[0:4]
j=9-crossover_point
for i in range(crossover_point):
    a.append(p[int(i+j)])
    b.append(s[int(i+j)])
    print(i)


print("Parents")
print("P1 :", a)
print("P2 :", b, "\n")

crossover_point = numpy.uint8(10/2)
print(crossover_point)

# # -----------

# offspring = []
# offspring_size=11
# crossover_point = offspring_size-4 # index 6

# p1 = [(62, 369), (296, 54), (319, 569), (454, 72), (120, 545), (264, 207), (418, 314), (98, 68), (271, 398), (24, 676), (66, 216)]
# p2 = [(132, 115), (265, 410), (39, 538), (338, 635), (444, 310), (18, 214), (352, 85), (190, 267), (405, 492), (173, 617), (76, 381)]

# cd1 = p1[0:crossover_point]
# cd2 = p2[0:crossover_point]

# j=crossover_point # 7
# for i in range(offspring_size-crossover_point):
# 	cd1.append(p2[int(i+j)])
# 	cd2.append(p1[int(i+j)])

# offspring.append(cd1)
# offspring.append(cd2)

# print("p1: ",p1)
# print("p2: ",p2)
# print("\n")
# print("o1: ",offspring[0])
# print("o2: ",offspring[1])

# test_list = [5, 6, 3, 7, 8, 1, 2, 10]
 
# test_list.pop(4)
# print(test_list)

# import cv2
# best_image = cv2.imread('imgAreaOutline0.png')
# cv2.imshow('result',best_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows