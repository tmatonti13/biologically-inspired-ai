


def fitness(genome): 
    ...
    return value 


def evaluate_fitness(population): 
    fitnesses = [fitness(g) for g in popoulation]
    avg_fitness = sum(fitnesses) / len(fitnesses)
    best_fitness = max(fitnesses)
    return avg_fitness, best_fitness

def select_pair(population): 
    sorted_genomes, sorted_fitnesses = sort_by_fitness(population)
    rank_values = [r for r in range(1, len(sorted_genomes)+1)]
    genome1 = weighted_choice(sorted_genomes,rank_values)
    genome2 = weighted_choice(sorted_genomes,rank_values)
    return genome1, genome2



def main(generations, population_size, crossover_rate, mutation_rate): 
    #create initial population
    population = make_population(population_size, genome_length)
    # calculate fitness value
    avg_fitness, best_fitness = evaluate_fitness(popualtion)
    for n in range(generations): 
        #build new population
        new_population = []
        for i in range(population_size // 2): 
            genome1, genome2 = select_pair(population)
            # perform crossover
            # perform mutation
            # add offspring to new population
        # replace old population with new population 
        population = new_population
        # calculate new fitness values
        avg_fitness, best_fitness = evaluate_fitness(population)