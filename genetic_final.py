#!/usr/bin/env python
# coding: utf-8

# In[1]:


################ Required libraries ##################
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the image in default color scale
image = cv2.imread('/home/vaasala/Desktop/img/top_terrain_1.jpg', 1)#import image in default colorscale
plt.imshow(image)#visulizing the image




# In[2]:


#################### Importing grayscale image ##################

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#conversion of image to grayscale
plt.imshow(gray_image)#visualizing grayscale image


# In[3]:


#################### grayscale histogram #############################

#array of histogram values of gray_image
histogram_a, bin_edges = np.histogram(gray_image, bins=256, range=(0, 256))
#bins=256 , the pixel values range from 0 to 255 which contain 256 pixel values
#range=(0,256), the range of the pixel values
plt.plot(histogram_a)#visualizing histogram


# In[112]:


################## Parameters ############################################

population_size = 100
generations = 200
thresholds_count = 5

################ Fitness function: Calculate between-class variance #######

def fitness(thresholds, gray_image):
    thresholds = sorted(thresholds) #sort the thresholds in increasing order
    thresholds = [0] + thresholds + [255] #Threshold list contains initial value 0 and final value 255
    total_variance = 0 #object that accumulates within class variance for each class
    
    for i in range(len(thresholds) - 1): #loops through each consecutive threshold
        lower, upper = thresholds[i], thresholds[i + 1] #consecutive thresholds
        mask = (gray_image >= lower) & (gray_image < upper) #Array of boolean values for the specified condition
        if np.sum(mask) == 0: #if the mask array values are NULL rejected
            continue
        class_mean = np.mean(gray_image[mask]) #extract only pixel values where mask array depicts true and calculate mean
        total_variance += np.sum(mask) * (class_mean - np.mean(gray_image)) ** 2 
        #np.sum(mask) = count of pixels which are true
        #np.mean(image)=total mean of all pixel values
        #class_mean=mean of a single class
    return total_variance #return the total between-class variance


############### Generate initial population ##############################

def initialize_population(population_size, thresholds_count):
    threshold_range=range(1,255)#range of the thresholds, pixel intensities
    population=[]#empty list to hold population
    for _ in range(population_size): #iterate through population size
        random_thresholds=random.sample(threshold_range,thresholds_count) 
        #generate random sample of threhsold values within range at the threshold count amount
        sorted_thresholds=sorted(random_thresholds) #sort thresholds in ascending order
        population.append(sorted_thresholds)#add sorted thresholds to the empty population list
    
    return population #return the population object




################ Selection: Roulette Wheel Selection ##################

def select(population, fitnesses):
    total_fitness = sum(fitnesses)#sum of fitness values in the chromosome
    probabilities=[]#initialize null object
    #iterating through the fitness values
    for f in fitnesses:
        probability=f/total_fitness
        probabilities.append(probability)
    num_individuals=len(population)#number of individuals in the population
    selected_index = np.random.choice(num_individuals, p=probabilities)#index selected at random based on probabilties p
    selected_individual = population[selected_index]#selected individual is the population values of selected index
    
    return selected_individual

################## Crossover ##########################################
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)#point where parents are divided into two
    #generates a random integer between two points
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    #addition of elements of parent1 elements upto crossover point to the parent2 elements beyond crossover point
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    #addition of parent2 elements upto crossover point to parent1 elements beyond crossover point
    
    return child1, child2

################## Mutation ###########################################
def mutate(thresholds):
    index = random.randint(0, len(thresholds) - 1) #selection of random position in the thresholds list for mutation
    thresholds[index] = random.randint(1, 254)#random integer selected within range introduced to index point
    
    return sorted(thresholds)

##################### Main Genetic Algorithm ##########################

def genetic_algorithm(gray_image, population_size, generations, thresholds_count):
    
    population = initialize_population(population_size, thresholds_count)#generating initial random population
    
    for generation in range(generations):
        
        fitnesses=[]
        for individual in population:
            fitness_score=fitness(individual, gray_image)
            fitnesses.append(fitness_score)
            
        new_population = []
        for _ in range(population_size // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])
        population = new_population
     
    #finding set of thresholds with highest fitness values
    best_individual = None #initialization
    best_fitness = -float('inf') #lowest fitness score as initialization
    #iterating over each individual in the population
    for ind in population:
        current_fitness = fitness(ind,gray_image) #calculating fitness score for currrent individual
        #updating best fitness score and best individual
        if current_fitness > best_fitness:
            best_fitness=current_fitness
            best_individual = ind #holds individual with highest fitness score
        
    
    
    return best_individual

##################### Run the genetic algorithm###########################

best_thresholds = genetic_algorithm(gray_image, population_size, generations, thresholds_count)


##################### Apply the best thresholds to the image ##############

thresholded_image = np.zeros_like(gray_image) #creating object with same shape as gray_image initialized to zero

thresholds = [0] + best_thresholds + [255]#initializing thresholds list

for i in range(len(thresholds) - 1):
    lower = thresholds[i] #lower bound for the thresholds
    upper = thresholds[i+1] #upper bound for the threesholds
    mask = (gray_image >= lower) & (gray_image < upper) #create mask for pixels within current interval
    midpoint = (lower + upper) // 2 #calculation of midpoint of thresholds
    thresholded_image[mask] = midpoint #applyind midpoint to mask of thresholded_image


##################### Display the result ###################################

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(gray_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Thresholded Image')
plt.imshow(thresholded_image, cmap='gray')
plt.show()


# In[113]:


####################### visulaizing thresholds ##############################
#Best thresholds
print(best_thresholds)
#grayscale histogram
histogram_b, bin_edges = np.histogram(thresholded_image, bins=256, range=(0, 256))
#bins=256 , the pixel values range from 0 to 255 which contain 256 pixel values
#range=(0,256), the range of the pixel values

#vertical lines for the thresholds
for threshold in best_thresholds:
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth='2', label=f'Thresholds')
    
plt.title('Histogram with best thresholds')
plt.plot(histogram_a)
plt.show


# In[114]:


################### PSNR value #########################
import numpy as np

def calculate_psnr(original, thresholded):
    # MSE calculation
    mse = np.mean((original - thresholded) ** 2)
    
    # PSNR calculation
    if mse == 0:
        return float('inf')  # Infinite PSNR if no error
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    
    return psnr

# Execution
psnr_value = calculate_psnr(gray_image, thresholded_image)
print("PSNR:", psnr_value)


# In[115]:


######################### SSIM ###########################
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original, thresholded):
    # SSIM computation
    ssim_value = ssim(original, thresholded, data_range=thresholded.max() - thresholded.min())
    
    return ssim_value

# Execution
ssim_value = calculate_ssim(gray_image, thresholded_image)
print("SSIM:", ssim_value)


# In[116]:


###################### execution time #######################
import time

def measure_execution_time(gray_image, population_size, generations, thresholds_count):
    # Start Timer
    start_time = time.time()
    
    # Run Genetic Algorithm
    best_thresholds = genetic_algorithm(gray_image, population_size, generations, thresholds_count)
    
    # Stop Timer
    end_time = time.time()
    
    # Execution Time calculation
    execution_time = end_time - start_time
    
    return execution_time

# Execution
execution_time = measure_execution_time(gray_image, population_size, generations, thresholds_count)
print("Execution Time:", execution_time, "seconds")


# In[117]:


################# dice coefficient #############################
def calculate_dice_coefficient(original, thresholded):
    # converting images to binary
    original_bin = original > 0  # Convert to binary based on a threshold
    thresholded_bin = thresholded > 0

    
    intersection = np.logical_and(original_bin, thresholded_bin).sum()
    union = original_bin.sum() + thresholded_bin.sum()

    # Dice coefficient computation
    dice_coefficient = (2.0 * intersection) / union
    
    return dice_coefficient

# Execution
dice_coefficient_value = calculate_dice_coefficient(gray_image, thresholded_image)
print("Dice Coefficient:", dice_coefficient_value)


# In[ ]:




