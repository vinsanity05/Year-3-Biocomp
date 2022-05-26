# 	Author: Vince Verdadero (19009246)
#   Created: 26 / 11 / 21
#   Last edited: 02 / 12 / 21
#   Description: A program to display the Counting ones maximisation function
#  	This program will:
#  	               Display the differences between the Roulette Wheel and Tournament selection
#                  Display a graph on 10 different runs to find out the average fit_value
#  	User advice: None

# These imports are used to help randomise my findings, copy one of the functions (deepcopy) and display the graphs

import random
import copy
import matplotlib.pyplot as plt


# A class called individual that stores of genes and fit_value values and represented as a data structure
class individual:
    gene = []  # Empty list of binary genes
    fit_value = 0  # Initialise the fitness value to 0

    # This is used to represent a class's objects as a string and will return the Gene and the value of the fitness
    def __repr__(self):
        return f"Gene string {''.join(str(x) for x in self.gene)} - Value_of_fitness: {str(self.fit_value)}"


# These are my parameters
Population_size = 50  # Random population size
number_of_genes = 50  # Random number of genes (This will be needed to test and would be altered for the experiment)
# down below)
Number_of_generations = 200  # initialise 200 generations (This will be needed to test and would be altered for the
# experiment down below)
Mutation_rate = 0.03  # Random Mutation Rate (This will be needed to test and would be  altered for the experiment
# down below)
Mutation_step = 0.9  # Random Mutation Step (This will be needed to test and would be altered for the experiment down
# below)

# This def function is to calculate the individual fitness
# This is also employing from Worksheet 3 that we have to do
def maximisation(individual):
    Value_of_fitness = 0
    for max in range(0, number_of_genes):
        Value_of_fitness = Value_of_fitness + individual.gene[max]
    return Value_of_fitness


#  For this def function, this will calculate the population's fitness
def fit_overall(population):
    calculating_fit = 0
    for individual in population:  # A for loop for every individual in the population
        calculating_fit += individual.fit_value
    return calculating_fit  # This will return to get the fitness total.


# For this def function, I am Initialising the original population

def original_pop():
    pop = []  # Empty list to store the population
    for n in range(0, Population_size):  # For loop to initialise the population
        short_term_gene = []  # This is a temporary gene list
        for n in range(0, number_of_genes):  # loop through the random genes length number
            short_term_gene.append(random.uniform(0.0, 1.0))
        new_individual = individual()
        new_individual.gene = short_term_gene.copy()
        new_individual.fit_value = maximisation(new_individual)
        pop.append(new_individual)  # Append the new individual to the population
    return pop


# This is a def function which is the tournament selection - This part is from Worksheet 1 but had to remove the deep
# copy since didn't work if added it on
def tournament(population):  # This will choose a fitter individual to pass their genes to the next generation:
    offspring = []  # Empty list for the offspring
    for x in range(0, Population_size):
        parent1 = random.randint(0, Population_size - 1)
        off_spring1 = population[parent1]
        parent2 = random.randint(0, Population_size - 1)
        off_spring2 = population[parent2]
        if not off_spring1.fit_value <= off_spring2.fit_value:
            offspring.append(off_spring1)
        else:
            offspring.append(off_spring2)

    return offspring


# This is a def function that i have implemented of Roulette Wheel selection
def roulette_wheel(population):
    # total fitness of original population
    original_fits = fit_overall(population)
    off_spring = copy.deepcopy(population)
    # This is the process of the roulette wheel
    for x in range(0, Population_size):
        RW_point = random.uniform(0.0, original_fits)
        overall_run = 0  # Initialise overall_run to 0
        r = 0  # Initialise r to 0
        while not overall_run > RW_point:
            overall_run += population[r].fit_value
            r += 1  # Incrementing one everytime
            if not r != Population_size:
                break
        off_spring[x] = population[r - 1]

    return off_spring


# This def function is the Crossover process
def crossover(offspring):
    cross_OS = []  # Empty list for the crossover offspring
    for x in range(0, Population_size, 2):
        cross_point = random.randint(0, number_of_genes - 1)  # picks a random cross_point in the gene length
        # Here we have two temporary and we stored as temporary individual instances
        short_term_1 = individual()
        short_term_2 = individual()
        # Here we have h1 and h2, both represent as heads, t1 and t2 represent tails
        h1 = []  # Empty list for head 1
        h2 = []  # Empty list for head 2
        t1 = []  # Empty list for tail 1
        t2 = []  # Empty list for tail 2
        for j in range(0, cross_point):
            h1.append(offspring[x].gene[j])  # adding gene and appending it to head1
            h2.append(offspring[x + 1].gene[j])  # adding gene and appending gene to head2
        for j in range(cross_point, number_of_genes):
            t1.append(offspring[x].gene[j])  # adding gene and appending gene to tail 1
            t2.append(offspring[x + 1].gene[j])  # adding gene and appending it to tail 2

        short_term_1.gene = h1 + t2  # add first gene after crossover to short_term_1
        short_term_2.gene = h2 + t1  # add second gene after crossover to short_term_2
        short_term_1.fit_value = maximisation(
            short_term_1)  # Calling to add fitness to the Counting ones short_term_1 individual
        short_term_2.fit_value = maximisation(
            short_term_2)  # Calling to add fitness to the Counting ones short_term_2 individual
        cross_OS.append(short_term_1)  # Appending the offspring crossover from the short_term_1
        cross_OS.append(short_term_2)  # Appending the offspring crossover from the short_term_2

    return cross_OS




# For this def function this will be the Bit-wise Mutation, this will mutate the result of new offspring - This is
# also from Worksheet 2

def mutation(offspring_crossover, mutation_rate, mutation_step):
    mutation_OS = []  # Empty list for mutation offspring
    for x in range(0, Population_size):
        new_individual = individual()
        new_individual.gene = []  # Empty list for new individual gene
        for y in range(0, number_of_genes):
            gene = offspring_crossover[x].gene[y]
            changing = random.uniform(0.0, mutation_step)  # This is to set 'changing' and randomise the mutation_step
            Mutation_probability = random.uniform(0.0, 100.0)
            if Mutation_probability < (100 * mutation_rate):
                if not random.randint(0, 1) != 1:  # if random num is 1 then it will increment to the 'changing'
                    gene += changing
                else:  # if random num is 0, then it will minus to the 'changing'
                    gene -= changing
                if gene > 1.0:  # if gene  is bigger than 1.0, then keep it at 1.0
                    gene = 1.0
                if gene < 0.0:  # if gene value is smaller than 0.0, then keep it at 0.0
                    gene = 0.0
            new_individual.gene.append(gene)  # This will append the gene from the new individual
        new_individual.fit_value = maximisation(
            new_individual)
        mutation_OS.append(new_individual)  # This will append the new_individual from the mutation offspring

    return mutation_OS


# For this def function I will be Descending and sort it based on the individual fitness
def descending(population):
    def sort(individual):
        return individual.fit_value

    population.sort(key=sort, reverse=True)

    return population


# This def function is using elitism for the Maximisation Optimisation
def elitism(population, new_population):
    population = descending(population)

    # Old best fit at indexes 0 and 1
    Old1_Best_fitness = population[0]
    Old2_Best_fitness = population[1]

    # Using the deepcopy for new population
    population = copy.deepcopy(new_population)

    population = descending(population)

    # Worst fit at index -1 and -2 in the new pop
    worstFit_new_1 = population[-1]
    worstFit_new_2 = population[-2]

    # This is to show if Old best fitness 1 and 2 is greater than new 1 and 2 worst fitness it will create a best
    # fitness for the maximisation
    if Old1_Best_fitness.fit_value > worstFit_new_1.fit_value:
        population[-1].fit_value = Old1_Best_fitness.fit_value
        population[-1].gene = Old1_Best_fitness.gene
    if Old2_Best_fitness.fit_value > worstFit_new_2.fit_value:
        population[-2].fit_value = Old2_Best_fitness.fit_value
        population[-2].gene = Old2_Best_fitness.gene

    return population


# In this def function, this will process everything together

def genetic_algorithm(population, selection, mutation_rate, mutation_step):
    # Global variables for the maximum and mean fitness
    global maximum_fitness, mean_fitness
    # storing data to plot
    values_for_mean_fitness = []  # Empty list for the mean fitness value
    values_for_maximum_fitness = []  # Empty list for the maximum fitness value

    for x in range(0, Number_of_generations):
        # tournament / Roulette Wheel selection process
        offspring = selection(population)
        # crossover process
        crossover_offspring = crossover(offspring)
        # mutation process
        mutate_offspring = mutation(crossover_offspring, mutation_rate, mutation_step)
        # This is the elitism process
        population = elitism(population, mutate_offspring)


        storing_fit = []  # Empty list to store the fitness
        for individual in population:
            storing_fit.append(maximisation(individual))

        maximum_fitness = max(storing_fit)  # take out the max fitness among the number of fitness in the storing_fit list
        mean_fitness = sum(storing_fit) / Population_size  # This is to calculate the mean fitness from the sum of the
        # fitness from the population size
        # append maxFit and meanFit respectively to MaxFit_values and MeanFit_values
        values_for_maximum_fitness.append(maximum_fitness)  # This is appending the maximum fitness from the value
        # maximum fitness list
        values_for_mean_fitness.append(mean_fitness)  # This is appending the mean fitness from the value mean
        # fitness list

    # This is to display what the mean and maximum fitness is from the output
    print(f"Maximum fitness: {str(maximum_fitness)}\n")
    print(f"Mean fitness: {str(mean_fitness)}\n")

    return values_for_maximum_fitness, values_for_mean_fitness


# plotting
plt.ylabel("Fitness")
plt.xlabel("Number of Generation")

# This to collect the data of maximum and mean fitness and put them into a list
data1_of_maximum_fitness = []
data2_of_maximum_fitness = []
data3_of_maximum_fitness = []
data4_of_maximum_fitness = []

data1_of_mean_fitness = []
data2_of_mean_fitness = []
data3_of_mean_fitness = []
data4_of_mean_fitness = []

# This is the testing stage and experimenting the selection methods for this
# maximisation

# ------------------------------------------------------------------------------

# In this section I will be comparing the tournament and roulette wheel selection

# This is the 1st test and will be experimented down below
# number_of_genes = 50

# These are the results given from the output from the 1st test
# Maximum Fitness: 50.0 - Tournament Selection
# Maximum Fitness: 49.68540419269904 - Roulette Wheel

# This is the 2nd test and will be experimented down below
# number_of_genes = 200 - - This what we will change
# Number_of_generations = 500 - This what we will change

# These are the results given from the output from the 2nd test
# Maximum Fitness: 196.66654480202897 - Tournament Selection
# Maximum Fitness: 177.7189295321322  - Roulette Wheel

# -------- Uncomment the code below for the 1st test  ---------------------

# number_of_genes = 50
# plt.title(
#     f"Maximisation genetic_algorithm \n Tournament and Roulette Wheel selection \nnumber of genes = "
#     f"{str(number_of_genes)} mutation rate = {str(Mutation_rate)} mutation step = {str(Mutation_step)}")
#
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.9)
# data2_of_maximum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="Tournament")
# plt.plot(data2_of_maximum_fitness, label="Roulette Wheel")


# -------- Uncomment the code below for the 2nd test  ---------------------

# number_of_genes = 200
# Number_of_generations = 500
#
# plt.title(
#     f"Maximisation genetic_algorithm \n Tournament and Roulette Wheel selection \nnumber of genes = "
#     f"{str(number_of_genes)} mutation rate = {str(Mutation_rate)} mutation step = {str(Mutation_step)}")
#
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.9)
# data2_of_maximum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="Tournament")
# plt.plot(data2_of_maximum_fitness, label="Roulette Wheel")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the tournament selection and finding the best fitness and mean
# fitness

# This is the 3rd test and will be experimented down below
# These are the results given from the output from the 3rd test
# Maximum Fitness: 50.0
# Mean Fitness: 49.56545039526755

# This is the 4th test and will be experimented down below
# number_of_genes = 200 - - This what we will change
# Number_of_generations = 500 - This what we will change

# These are the results given from the output from the 4th test
# Maximum Fitness: 196.35731143850094
# Mean Fitness: 192.7203254790771

# -------- Uncomment the code below for 3rd test  ---------------------

# plt.title(
#     f"Maximisation genetic_algorithm - Tournament selection \nnumber of genes = {str(number_of_genes)} mutation rate = "
#     f"{str(Mutation_rate)} mutation step = {str(Mutation_step)}")
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="Maximum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# -------- Uncomment the code below for 4th test  ---------------------

# number_of_genes = 200
# Number_of_generations = 500
#
# plt.title(
#     f"Maximisation genetic_algorithm - Tournament selection \nnumber of genes = {str(number_of_genes)} mutation rate = "
#     f"{str(Mutation_rate)} mutation step = {str(Mutation_step)}")
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="Maximum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation rate and experimenting those tests to see which
# mutation rate increased the most for fitness for the tournament selection

# This is the 5th test and will be experimented down below

# These are the results given from the output from the 5th test and as you can see 0.03 is the best for increasing fitness
# Maximum Fitness: 44.723637947677396 - mutation_rate 0.3
# Maximum Fitness: 50.0 - mutation_rate 0.03
# Maximum Fitness: 49.981995699791014 - mutation_rate 0.003
# Maximum Fitness: 43.659661242850696 - mutation_rate 0.0003

# -------- Uncomment the code below for 5th test  ---------------------

# plt.title("Maximisation genetic algorithm - Tournament selection \nVary mutation rate")
#
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.3, 0.9)
# data2_of_maximum_fitness, data2_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.9)
# data3_of_maximum_fitness, data3_of_mean_fitness = genetic_algorithm(population, tournament, 0.003, 0.9)
# data4_of_maximum_fitness, data4_of_mean_fitness = genetic_algorithm(population, tournament, 0.0003, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="mutation rate 0.3")
# plt.plot(data2_of_maximum_fitness, label="mutation rate 0.03")
# plt.plot(data3_of_maximum_fitness, label="mutation rate 0.003")
# plt.plot(data4_of_maximum_fitness, label="mutation rate 0.0003")

# ------------------------------------------------------------------------------
# For this section, I will be conducting a test on the vary mutation step and experimenting those tests to see which
# mutation step increased the most for fitness for the Tournament selection

# This is the 6th test and will be experimented down below
# These are the results given from the output from the 6th test and as you can see there is no best one it since it all stays at 50
# Maximum Fitness: 50.0 - mutation step 0.3
# Maximum Fitness: 50.0 - mutation step 0.9
# Maximum Fitness: 50.0 - mutation step 0.6
# Maximum Fitness: 50.0 - mutation step 1.0

#However in terms to find the best one - the average/mean of the best between this lot are listed below and 0.3 is the
# best for mutation step for increasing fitness
# Mean Fitness: 49.85429837352004 - mutation step 0.3
# Mean Fitness: 49.4310628355658 - mutation step 0.9
# Mean Fitness: 49.57862350690878 - mutation step 0.6
# Mean Fitness: 49.425511079432766 - mutation step 1.0

# -------- Uncomment the code below for the 6th test  ---------------------

# plt.title("Maximisation genetic algorithm - Tournament selection \nVary mutation step")
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.3)
# data2_of_maximum_fitness, data2_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.9)
# data3_of_maximum_fitness, data3_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.6)
# data4_of_maximum_fitness, data4_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 1.0)
#
# plt.plot(data1_of_maximum_fitness, label="mutation step 0.3")
# plt.plot(data2_of_maximum_fitness, label="mutation step 0.9")
# plt.plot(data3_of_maximum_fitness, label="mutation step 0.6")
# plt.plot(data4_of_maximum_fitness, label="mutation step 1.0")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the Roulette Wheel selection and conducting the best fitness and
# the mean fitness for this Roulette Wheel selection

# This is the 7th test and will be experimented down below

# These are the results given from the output from the 7th test
# Maximum Fitness: 49.73681692881439
# Mean Fitness: 47.356735736067165

# This is the 8th test and will be experimented down below

# number_of_genes = 200 - This is what we will change
# Number_of_generations = 500 - This is what we will change

# These are the results given from the output from the 8th test
# Maximum Fitness: 177.68517124750394
# Mean Fitness: 167.3043662854009

# -------- Uncomment the code below for 7th test  ---------------------

# plt.title(
#     f"Maximisation genetic algorithm - Roulette Wheel selection \nnumber of genes = {str(number_of_genes)} mutation "
#     f"rate = {str(Mutation_rate)} mutation step = {str(Mutation_step)}")
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="Maximum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# -------- Uncomment the code below for 8th test  ---------------------

# number_of_genes = 200
# Number_of_generations = 500
#
# plt.title(
#     f"Maximisation genetic algorithm - Roulette Wheel selection \nnumber of genes = {str(number_of_genes)} mutation "
#     f"rate = {str(Mutation_rate)} mutation step = {str(Mutation_step)}")
#
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="Maximum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation rate and experimenting those tests to see which
# mutation rate increased the most for fitness for the Roulette Wheel selection

# This is the 9th test and will be experimented down below
# These are the results given from the output from the 9th test and as you can see 0.03 is the best for increasing fitness
# Maximum Fitness: 43.48467490475574 - mutation_rate 0.3
# Maximum Fitness: 48.791926742317315 - mutation_rate 0.03
# Maximum Fitness: 47.95331056719819 - mutation_rate 0.003
# Maximum Fitness: 42.56852059163248 - mutation_rate 0.0003

#  --- Uncomment the code below for 9th test  ----------------------

# plt.title("Maximisation genetic algorithm - Roulette Wheel selection \nVary mutation rate")
#
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.3, 0.9)
# data2_of_maximum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.9)
# data3_of_maximum_fitness, data3_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.003, 0.9)
# data4_of_maximum_fitness, data4_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.0003, 0.9)
#
# plt.plot(data1_of_maximum_fitness, label="mutation rate 0.3")
# plt.plot(data2_of_maximum_fitness, label="mutation rate 0.03")
# plt.plot(data3_of_maximum_fitness, label="mutation rate 0.003")
# plt.plot(data4_of_maximum_fitness, label="mutation rate 0.0003")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation step and experimenting those tests to see which
# mutation step increased the most for fitness for the Roulette Wheel selection

# This is the 10th test and will be experimented down below
# These are the results given from the output from the 10th test and as you can see 0.9 is the best for increasing
# fitness
# Maximum Fitness: 48.275278639417465 - mutation step 0.3
# Maximum Fitness: 49.86481084718186 - mutation step 0.9
# Maximum Fitness: 49.4756549983502 - mutation step 0.6
# Maximum Fitness: 49.7304374102101 - mutation step 1.0


# -------- Uncomment the code below to for 10th test  ---------------------
#
# plt.title("Maximisation genetic algorithm - Roulette Wheel selection  \nVary mutation step")
#
#
# population = original_pop()
#
# data1_of_maximum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.3)
# data2_of_maximum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.9)
# data3_of_maximum_fitness, data3_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.6)
# data4_of_maximum_fitness, data4_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)
#
# plt.plot(data1_of_maximum_fitness, label="mutation step 0.3")
# plt.plot(data2_of_maximum_fitness, label="mutation step 0.9")
# plt.plot(data3_of_maximum_fitness, label="mutation step 0.6")
# plt.plot(data4_of_maximum_fitness, label="mutation step 1.0")

# ------------------------------------------------------------------------------


# This is to display the location of the plots (this is to used to determine what line is which)
plt.legend(loc="lower right")
plt.show()
