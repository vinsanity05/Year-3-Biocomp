# 	Author: Vince Verdadero (19009246)
#   Created: 26 / 11 / 21
#   Last edited: 06 / 12 / 21
#   Description: A program to display the Rosenbrock minimisation function
#  	This program will:
#  	               Display the differences between the Roulette Wheel and Tournament selection
#                  Display a graph on 10 different runs to find out the average/mean fitness
#  	User advice: None

# These imports are used to help randomise my findings, help with writing my minimisation function and display the
# graphs

import random
import copy
import math

import matplotlib.pyplot as plt


# A class called individual that stores the genes and fit_value values and represented as a data structure
class Individual:
    gene = []  # Empty list of binary genes
    fit_value = 0  # Initialise the fitness value value to 0

    # This is used to represent a class's objects as a string and will return the Gene and the value of the fitness
    def __repr__(self):
        return f"Gene string {''.join(str(x) for x in self.gene)} - Value_of_fitness: {str(self.fit_value)}"


# These are my parameters
Population_size = 50  # Random population number
number_of_genes = 10  # Random number of genes (This will be needed to test and would be altered for the experiment
# down below)
Number_of_generations = 500  # initialise 500 generations (This will be needed to test and would be altered for the
# experiment down below)
Mutation_rate = 0.03  # Random Mutation Rate (This will be needed to test and would be  altered for the experiment
# down below)
Mutation_step = 1.0  # Random Mutation Step (This will be needed to test and would be altered for the experiment down
# below)


# This def function which is the Rosenbrock minimisation function  will Calculate the individual's fitness
def rosenbrock_minimisation_function(individual):
    for i in range(0, number_of_genes):
        sum1 = individual.gene[i]
        sum2 = individual.gene[i + 1]
        fitness = (100 * (sum2 - sum1 ** 2)) ** 2 + (1 - sum1) ** 2
        return fitness


#  For this def function, this will calculate the population's fit_value
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
            # a random gene between -100 and 100  - # this is from the task we have to implement
            short_term_gene.append(random.uniform(-100.0, 100.0))
        new_individual = Individual()
        new_individual.gene = short_term_gene.copy()
        new_individual.fit_value = rosenbrock_minimisation_function(new_individual)
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
            offspring.append(off_spring2)
        else:
            offspring.append(off_spring1)

    return offspring


# This is a def function that i have implemented of Roulette Wheel selection
def roulette_wheel(population):
    total_fit_initial_pop = 0  # total fitness of original pop
    for individual in population:
        total_fit_initial_pop += 1 / individual.fit_value
    offspring = []  # Empty list for the offspring
    # This is the process of the roulette wheel
    for x in range(0, Population_size):
        RW_point = random.uniform(0.0, total_fit_initial_pop)
        overall_run = 0  # Initialise overall_run to 0
        r = 0  # Initialise r to 0
        while overall_run <= RW_point:
            overall_run += 1 / population[r].fit_value
            r += 1  # Incrementing one everytime
            if not r != Population_size:
                break
        offspring.append(
            copy.deepcopy(population[r - 1]))

    return offspring


# This def function is the Crossover process
def crossover(offspring):
    cross_OS = []  # Empty list for the crossover offspring
    for x in range(0, Population_size, 2):
        cross_point = random.randint(1, number_of_genes - 1)  # picks a random cross_point in the gene length
        # Here we have two temporary and we stored as temporary individual instances
        short_term1 = Individual()
        short_term2 = Individual()
        # Here we have h1 and h2, both represent as heads, t1 and t2 represent tails
        h1 = []  # Empty list for head 1
        h2 = []  # Empty list for head 2
        t1 = []  # Empty list for tail 1
        t2 = []  # Empty list for tail 2
        for j in range(0, cross_point):
            h1.append(offspring[x].gene[j])  # adding gene and appending it to head1
            h2.append(offspring[x + 1].gene[j])  # adding gene and appending gene to head2
        for j in range(cross_point, number_of_genes):
            t1.append(offspring[x].gene[j])  # adding gene and appending gene to tail1
            t2.append(offspring[x + 1].gene[j])  # adding gene and appending it to tail2

        short_term1.gene = h1 + t2  # add first gene after crossover to short_term1
        short_term2.gene = h2 + t1  # add second gene after crossover to short_term2
        short_term1.fit_value = rosenbrock_minimisation_function(
            short_term1)  # Calling to add fit_value to the minimisation short_term1 individual
        short_term2.fit_value = rosenbrock_minimisation_function(
            short_term2)  # Calling to add fit_value to the minimisation short_term2 individual
        cross_OS.append(short_term1)  # Appending the offspring crossover from the temporary 1
        cross_OS.append(short_term2)  # Appending the offspring crossover from the temporary 2

    return cross_OS


# For this def function this will be the Bit-wise Mutation, this will mutate the result of new offspring - This is
# also from Worksheet 2

def mutation(offspring_crossover, mutation_rate, mutation_step):
    mutation_OS = []  # Empty list for mutation offspring
    for x in range(0, Population_size):
        new_individual = Individual()
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
                if gene > 100.0:  # if the gene is bigger than 100, then keep it to 100
                    gene = 100
                if gene < -100.0:  # if the gene is smaller than -100, then keep it to -100
                    gene = -100
            new_individual.gene.append(gene)  # This will append the gene from the new individual
        new_individual.fit_value = rosenbrock_minimisation_function(
            new_individual)
        mutation_OS.append(new_individual)  # This will append the new_individual from the mutation offspring

    return mutation_OS


# For this def function I will be Descending and sort it based on the individual fitness
def descending(population):
    def sort(individual):
        return individual.fit_value

    population.sort(key=sort, reverse=True)

    return population


# This def function is using elitism for the Minimisation Optimisation
def elitism(population, new_population):
    population = descending(population)

    # Old worst fit at indexes -1 and 2
    Old1_Worst_fitness = population[-1]
    Old2_Worst_fitness = population[-2]

    # Using the deepcopy for new population
    population = copy.deepcopy(new_population)

    population = descending(population)

    # Best fit at index 0 and 1 in the new pop
    new1_Best_fitness = population[0]
    new2_Best_fitness = population[1]

    # This is to show if Old worst fitness 1 and 2 is less than new 1 and 2 best fitness it will create a worst
    # fitness for the minimisation
    if Old1_Worst_fitness.fit_value < new1_Best_fitness.fit_value:
        population[0].gene = Old1_Worst_fitness.gene
        population[0].fit_value = Old1_Worst_fitness.fit_value
    if Old2_Worst_fitness.fit_value < new2_Best_fitness.fit_value:
        population[1].gene = Old2_Worst_fitness.gene
        population[1].fit_value = Old2_Worst_fitness.fit_value

    return population


# In this def function, this will process everything together

def genetic_algorithm(population, selection, mutation_rate, mutation_step):
    # Global variables for the minimum and mean fitness
    global minimum_fitness, mean_fitness
    # These are to plot and to collect data
    values_for_mean_fitness = []  # Empty list for the mean fitness value
    values_for_minimum_fitness = []  # Empty list for the minimum fitness value

    for x in range(0, Number_of_generations):
        # tournament / Roulette Wheel selection process
        offspring = selection(population)
        # crossover process
        crossover_offspring = crossover(offspring)
        # mutation process
        mutate_offspring = mutation(crossover_offspring, mutation_rate, mutation_step)
        # This is the elitism process
        population = elitism(population, mutate_offspring)

        # calculate Min and Mean Fitness
        storing_fit = []  # Empty list to store the fitness
        for individual in population:
            storing_fit.append(rosenbrock_minimisation_function(individual))

        minimum_fitness = min(
            storing_fit)  # take out the min fit_value among the number of fitness in the storing_fit list
        mean_fitness = sum(
            storing_fit) / Population_size  # This is to calculate the mean fitness from the sum of the
        # fitness from the  population size

        values_for_minimum_fitness.append(
            minimum_fitness)  # This is appending the minimum fitness from the value minimum
        # fitness list
        values_for_mean_fitness.append(
            mean_fitness)  # This is appending the mean fitness from the value mean fitness list

        # This is to display what the mean and minimum fit_value is from the output
    print(f"Minimum Fitness: {str(minimum_fitness)}\n")
    print(f"Mean Fitness: {str(mean_fitness)}\n")

    return values_for_minimum_fitness, values_for_mean_fitness


# plotting
plt.ylabel("Fitness")
plt.xlabel("Number of Generation")

# This to collect the data of minimum and mean fit_value and put them into a list
data1_of_minimum_fitness = []
data2_of_minimum_fitness = []
data3_of_minimum_fitness = []
data4_of_minimum_fitness = []

data1_of_mean_fitness = []
data2_of_mean_fitness = []
data3_of_mean_fitness = []
data4_of_mean_fitness = []

# This is the testing stage and experimenting the comparison between roulette wheel and tournament selection for this
# minimisation
# ------------------------------------------------------------------------------
# In this section I will be comparing the tournament and roulette wheel selection

# This is the 1st test and will be experimented down below
# number_of_genes = 10
# Number_of_generations = 500

# These are the results given from the output from the 1st test
# Minimum Fitness: 0.04675685781301373 - This will be output  the Tournament selection
# Minimum Fitness: 3.0513237789938987 - This will be output the Roulette Wheel selection

# This is the 2nd test and will be experimented down below
# number_of_genes = 20 - This is what we will change
# mutation_rate = 0.03
# mutation_step = 1.0
# Number_of_generations = 2000 - This what we will change

# These are the results given from the output from the 2nd test
# Minimum Fitness: 0.6675311201741557 - This will be output  the Tournament selection
# Minimum Fitness: 0.7652588513390385 - This will be output  the Roulette Wheel selection

# -------- Uncomment the code below for the 1st test  ---------------------

number_of_genes = 10
Number_of_generations = 500
plt.title(
    f"Minimisation Genetic algorithm \n Tournament and Roulette Wheel selection \nNumber of genes = "
    f"{str(number_of_genes)} Mutation rate = {str(Mutation_rate)} Mutation step = {str(Mutation_step)}")

# initialise original population
population = original_pop()

data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 1.0)
data2_of_minimum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)

plt.plot(data1_of_minimum_fitness, label="Tournament selection")
plt.plot(data2_of_minimum_fitness, label="Roulette Wheel selection")

# -------- Uncomment the code below for the 2nd test  ---------------------

# number_of_genes = 20
# Number_of_generations = 2000
# plt.title(
#     f"Minimisation Genetic algorithm \n Tournament and Roulette Wheel selection \nNumber of genes = "
#     f"{str(number_of_genes)} Mutation rate = {str(Mutation_rate)} Mutation step = {str(Mutation_step)}")
#
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 1.0)
# data2_of_minimum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Tournament selection")
# plt.plot(data2_of_minimum_fitness, label="Roulette Wheel selection")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the tournament selection and finding the best fitness and mean
# fitness

# This is the 3rd test and will be experimented down below
# These are the results given from the output from the 3rd test
# Minimum Fitness: 0.47955751554643433
# Mean Fitness: 0.47955751554643455

# This is the 4th test and will be experimented down below
# number_of_genes = 20 - This is what we will change
# Number_of_generations = 2000 - This what we will change

# These are the results given from the output from the 4th test
# Minimum Fitness: 0.2908852330233489
# Mean Fitness: 0.2908852330233492


# -------- Uncomment the code below for 3rd test ---------------------

# plt.title(
#     f"Minimisation Genetic Algorithm - Tournament selection \nNumber of genes = {str(number_of_genes)} Mutation rate = "
#     f"{str(Mutation_rate)} Mutation step = {str(Mutation_step)}")
#
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Minimum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# -------- Uncomment the code below for 4th test  ---------------------

# number_of_genes = 20
# Number_of_generations = 2000
# plt.title(
#     f"Minimisation Genetic Algorithm - Tournament selection \nNumber of genes = {str(number_of_genes)} Mutation rate = "
#     f"{str(Mutation_rate)} Mutation step = {str(Mutation_step)}")
#
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Minimum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation rate and experimenting those tests to see which
# mutation rate decreased the most for fitness for the tournament selection

# This is the 5th test and will be experimented down below

# These are the results given from the output from the 5th test and as you can see 0.03 is the best for decreasing fitness
# Minimum Fitness: 7.45778406198979 - mutation_rate 0.3
# Minimum Fitness: 3.7670108478633475 - mutation_rate 0.03
# Minimum Fitness: 1542911.9711373826 - mutation_rate 0.003
# Minimum Fitness: 158772.34586930278 - mutation_rate 0.0003


# -------- Uncomment the code below for 5th test  ---------------------


# plt.title("Minimisation Genetic Algorithm - Tournament selection \nVary Mutation rate")
#
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.3, 1.0)
# data2_of_minimum_fitness, data2_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 1.0)
# data3_of_minimum_fitness, data3_of_mean_fitness = genetic_algorithm(population, tournament, 0.003, 1.0)
# data4_of_minimum_fitness, data4_of_mean_fitness = genetic_algorithm(population, tournament, 0.0003, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Mutation rate 0.3")
# plt.plot(data2_of_minimum_fitness, label="Mutation rate 0.03")
# plt.plot(data3_of_minimum_fitness, label="Mutation rate 0.003")
# plt.plot(data4_of_minimum_fitness, label="Mutation rate 0.0003")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation step and experimenting those tests to see which
# mutation step decreased the most for fitness for the Tournament selection

# This is the 6th test and will be experimented down below

# These are the results given from the output from the 6th test and as you can see 1.0 is the best for decreasing fitness
# Minimum Fitness: 0.7783590526885409 - mutation_step 1.0
# Minimum Fitness: 66.88658099218682 - mutation_step 0.3
# Minimum Fitness: 66.88848858492034 - mutation_step 0.5
# Minimum Fitness: 66.88755895279033 - mutation_step 0.8

# -------- Uncomment the code below for the 6th test ---------------------

# plt.title("Minimisation Genetic Algorithm - Tournament selection \nVary Mutation step")
#
# #initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, tournament, 0.03,  1.0)
# data2_of_minimum_fitness, data2_of_mean_fitness = genetic_algorithm(population, tournament, 0.03,  0.3)
# data3_of_minimum_fitness, data3_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.5)
# data4_of_minimum_fitness, data4_of_mean_fitness = genetic_algorithm(population, tournament, 0.03, 0.8)
#
# plt.plot(data1_of_minimum_fitness, label="Mutation step 1.0")
# plt.plot(data2_of_minimum_fitness, label="Mutation step 0.3")
# plt.plot(data3_of_minimum_fitness, label="Mutation step 0.5")
# plt.plot(data4_of_minimum_fitness, label="Mutation step 0.8")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the Roulette Wheel selection and conducting the best fitness and
# the mean fitness for this Roulette Wheel selection

# This is the 7th test and will be experimented down below

# These are the results given from the output from the 7th test
# Minimum Fitness: 0.5152251383208061
# Mean Fitness: 0.5152251383208057

# This is the 8th test and will be experimented down below
# number_of_genes = 20 - This is what we will change
# Number_of_generations = 2000 - This is what we will change

# These are the results given from the output from the 8th test
# Minimum Fitness: 0.8612633528856898
# Mean Fitness: 0.8612633528856907

# -------- Uncomment the code below for 7th test  ---------------------

# plt.title(
#     f"Minimisation Genetic Algorithm - Roulette Wheel selection \nNumber of genes = {str(number_of_genes)} Mutation rate"
#     f" = {str(Mutation_rate)} Mutation step = {str(Mutation_step)}")
#
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Minimum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# -------- Uncomment the code below for 8th test  ---------------------

# number_of_genes = 20
# Number_of_generations = 2000
# plt.title(
#     f"Minimisation Genetic Algorithm - Roulette Wheel selection \nNumber of genes = {str(number_of_genes)} Mutation rate"
#     f" = {str(Mutation_rate)} Mutation step = {str(Mutation_step)}")
#
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Minimum Fitness")
# plt.plot(data1_of_mean_fitness, label="Mean Fitness")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation rate and experimenting those tests to see which
# mutation rate decreased the most for fitness for the Roulette Wheel selection

# This is the 9th test and will be experimented down below

# These are the results given from the output from the 9th test and as you can see 0.03 is the best for decreasing fitness
# Minimum Fitness: 11.51301614919247 - mutation_rate 0.3
# Minimum Fitness: 11.43437092882608 - mutation_rate 0.03
# Minimum Fitness: 14.186172901198361 - mutation_rate 0.003
# Minimum Fitness: 410.56423037045363 - mutation_rate 0.0003


#  --- Uncomment the code below for 9th test ----------------------
#
# plt.title("Minimisation Genetic Algorithm - Roulette Wheel selection \nVary Mutation Rate")
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.3, 1.0)
# data2_of_minimum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)
# data3_of_minimum_fitness, data3_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.003, 1.0)
# data4_of_minimum_fitness, data4_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.0003, 1.0)
#
# plt.plot(data1_of_minimum_fitness, label="Mutation Rate 0.3")
# plt.plot(data2_of_minimum_fitness, label="Mutation Rate 0.03")
# plt.plot(data3_of_minimum_fitness, label="Mutation Rate 0.003")
# plt.plot(data4_of_minimum_fitness, label="Mutation Rate 0.0003")

# ------------------------------------------------------------------------------

# For this section, I will be conducting a test on the vary mutation step and experimenting those tests to see which
# mutation step decreased the most for fitness for the Roulette Wheel selection

# This is the 10th test and will be experimented down below

# These are the results given from the output from the 10th test and as you can see 1.0 is the best for decreasing fitness
# Minimum Fitness: 1.0390480354380542 - mutation_step 1.0
# Minimum Fitness: 1448542.1358728495 - mutation_step 0.3
# Minimum Fitness: 97382.48375883694 - mutation_step 0.5
# Minimum Fitness: 2.173148964654722 - mutation_step 0.8


# -------- Uncomment the code below to for 10th test  ---------------------

# plt.title("Minimisation Genetic Algorithm - Roulette Wheel selection \nVary Mutation Step")
# # initialise original population
# population = original_pop()
#
# data1_of_minimum_fitness, data1_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 1.0)
# data2_of_minimum_fitness, data2_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.3)
# data3_of_minimum_fitness, data3_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.5)
# data4_of_minimum_fitness, data4_of_mean_fitness = genetic_algorithm(population, roulette_wheel, 0.03, 0.8)
#
# plt.plot(data1_of_minimum_fitness, label="Mutation Step 1.0")
# plt.plot(data2_of_minimum_fitness, label="Mutation Step 0.3")
# plt.plot(data3_of_minimum_fitness, label="Mutation Step 0.5")
# plt.plot(data4_of_minimum_fitness, label="Mutation Step 0.8")

# ------------------------------------------------------------------------------

# This is to display the location of the plots (this is to used to determine what line is which)
plt.legend(loc="upper right")
plt.show()
