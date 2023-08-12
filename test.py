# initial_population = [1,2,3,4,5,6,7,8]
# parents = [11,22]
# offspring_mutation = [88,99]

# current_population = initial_population

# current_population[-4:-2] = parents[-2:]
# current_population[-2:] = offspring_mutation[-2:]

# print(current_population)

# value = []

# def LastNlines(fname, N):
# 	# opening file using with() method
# 	# so that file get closed
# 	# after completing work
#     with open(fname) as file:
#         for line in file:
#             pass
#         last_line = line

#         # print(last_line)

#     return last_line

# 	# with open(fname) as file:
		
# 	# 	# loop to read iterate
# 	# 	# last n lines and print it
# 	# 	for line in (file.readlines() [-N:]):
# 	# 		print(line, end ='')

# # Driver Code:
# if __name__ == '__main__':
# 	fname = 'File1.txt'
# 	N = 2
# 	try:
# 		LastNlines(fname, N)
# 	except:
# 		print('File not found')
                
# print(LastNlines(fname, N))

# idx =  [(256, 295), (188, 179), (268, 75), (96, 425), (308, 140), (117, 327), (323, 313), (295, 202)]
# for x in idx:
#       if x != idx[5] and x != idx[6]:
#             print(x)
            
# idx[5] = (1234,3455)
# print(idx)

def last_two_min_indices(lst):
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])
    last_two_indices = sorted_indices[:2]
    return last_two_indices

# Example usage:
my_list = [10, 5, 8, 3, 12, 7]
result = last_two_min_indices(my_list)
print("Indices of the last two minimum values:", result)

# best_chr_fitness = 0.350183250923
# previous_fitness = 0.232576294324

# convergence = abs(best_chr_fitness - previous_fitness)

# print(convergence)

# mylist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']
# def rt (lis): 
# 	first4 = lis[:2] 
# 	last4 = lis[-2:] 
# 	first4.extend(last4) 
# 	return (first4)

# print(rt(mylist))

# for i in range(4):
# 	print(i)

fitness_values = [12321.421, 1321421.3213, 13124.31231, 123124.2141, 12.432, 312.432, 321345,412341, 4124,534135,52345234,35113]

li=[]
 
for i in range(len(fitness_values)):
      li.append([fitness_values[i],i])
li.sort()
sort_index = []
 
for x in li:
      sort_index.append(x[1])
 
print(sort_index)

last_two_indices = sort_index[:2]
first_two_indices = sort_index[-2:]
indices = last_two_indices + first_two_indices
print(indices)

# sorted_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])
# last_two_indices = sorted_indices[:2]
# first_two_indices = sorted_indices[-2:]
# indices = first_two_indices.extend(last_two_indices)
# print(indices)