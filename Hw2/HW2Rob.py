import robby 
import random
import numpy as np 
import matplotlib.pyplot as plt 

#graphics off, so the program can be faster 
# beginning of GA, then can turn it on after 
# find average of all 

# use approach from first assignment as the basis for this one
# genome has to be 243 strings 
# Fitness function has to simulate robby for 100 sessions by 200 steps 
# selection changes 

genome_length = 243 
Actions = ["MoveNorth", "MoveSouth", "MoveEast", "MoveWest", "StayPut", "PickUpCan","MoveRandom"]
rw = robby.World(10,10)



def evaluate_fitness(population): 
    sorted_genomes, sorted_fitnesses = sort_by_fitness(population)
    fitnesses = [fitness(g) for g in population]
    avg_fitness = sum(sorted_fitnesses) / len(population)
    best_fitness = sorted_fitnesses[-1]
    return avg_fitness, best_fitness, sorted_genomes


def random_genome(length=243):
    genome = "" 
    for i in range(length):
        genome += str(random.choice('0123456'))
    return genome

def make_population(size, length): 
    population = [] 
    for i in range(size): 
        genome = random_genome(243) 
        population.append(genome)
    return population

def crossover(genome1, genome2): 
    assert len(genome1) == len(genome2) == 243 
    cross = random.randint(1, len(genome1) - 1)
    cross1 = genome1[:cross] + genome2[cross:]
    cross2 = genome2[:cross] + genome1[cross:]
    return cross1, cross2 

def mutate(genome, mutation_rate): 
    new = []
    for genes in genome: 
        if random.random() < mutation_rate: 
            new.append(random.choice("0123456"))
        else: 
            new.append(genes)
    return "".join(new)


#uses rank selection 
def select_pair(sorted_genomes, rank_values): 
    genome1 = weighted_choice(sorted_genomes,rank_values)
    genome2 = weighted_choice(sorted_genomes,rank_values)
    return genome1, genome2


def sort_by_fitness(population):
    tuples = [(fitness(g), g) for g in population]
    tuples = sorted(tuples)
    sorted_fitness_values = [f for (f, g) in tuples]
    sorted_genomes = [g for (f, g) in tuples]
    return sorted_genomes, sorted_fitness_values


def weighted_choice(elements, weights):
        assert len(elements) == len(weights), "mismatched list lengths"
        for w in weights:
            assert w >= 0, "negative weights not allowed" 
        total = sum(weights)
        choice = random.uniform(0,total)
        current = 0
        for i in range(0, len(elements)):
            current += weights[i]
            if current > choice:
                return elements[i]
            


def move_robby(genome): 
    location = rw.get_percept_code()
    g_index = int (genome[location])
    action_index = Actions[g_index]
    reward = rw.perform_action(action_index)
    return reward 


def fitness(genome): 
    total = 0 
    session_total = 0
    rw.goto(0,0)
    rw.distribute_cans(0.5)
    for i in range(200): 
        session_total += move_robby(genome)
        total+= session_total
    return total / 100


def main(generations, population_size, crossover_rate, mutation_rate, logfile=""): 
    rw.graphics_off("Evaluating Fitness")
    #create initial population
    population = make_population(population_size, genome_length)
    rank_values = [r for r in range(1, population_size+1)]
    # calculate fitness value
    avg_fitness, best_fitness, sorted_genomes = evaluate_fitness(population)
    for n in range(generations): 
        #build new population
        new_population = []
        for i in range(population_size // 2): 
            genome1, genome2 = select_pair(sorted_genomes, rank_values)
            # crossover
            if random.random() < crossover_rate:
                child1, child2 = crossover(genome1, genome2)
            else:
                child1, child2 = genome1, genome2
            # perform mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            # add offspring to new population
            new_population.append(child1)
            new_population.append(child2)
        # replace old population with new population 
        population = new_population
        # calculate new fitness values
        avg_fitness, best_fitness, sorted_genomes = evaluate_fitness(population)
        print(f"{avg_fitness}, {best_fitness}")
        if logfile != "" and n%10 ==0:
            with open(logfile, "a") as f:
                f.write(f"{n} {avg_fitness} {best_fitness} {sorted_genomes[-1]} \n") 


main(500, 400, 1.0, 0.01, "BestFitness3.txt") 


# Trial 1:  main(500, 200, 1.0, 0.005, "GAoutput.txt") = -6.6670500000000015, 172.8





