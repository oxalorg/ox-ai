"""
The simplest program I can come up with based on the
limited knowledge I have about genetic algorithms
"""
import logging
import bisect
import random
import sys
import itertools


class NQueen():
    def __init__(self, N=8):
        self.size = N
        self.max_fitness = self.size * (self.size - 1) // 2

    def fitness(self, sol):
        """sol is a list with row number of ith column eg. [2341678]"""
        assert len(sol) == self.size
        sol = [int(i) for i in sol]
        a_pairs = 0  # attacking pairs
        for queen in range(self.size):
            for ahead_queen in range(queen + 1, self.size):
                # each queen ahead of current queen
                if sol[queen] == sol[ahead_queen]:
                    # same row
                    a_pairs += 1
                if abs(sol[ahead_queen] - sol[queen]) == ahead_queen - queen:
                    # diagonals
                    a_pairs += 1

        na_pairs = self.max_fitness - a_pairs
        return na_pairs


class GeneticAlgorithm():
    def __init__(self, pop_size, fitness_fn, max_fitness):
        self.prob_mutation = 0.03
        self.pop_size = pop_size
        self.fitness = fitness_fn
        self.max_fitness = max_fitness
        self.seq = [i for i in range(1, pop_size + 1)]

    def ga(self):
        """Genetic algorithm"""
        # Generate an initial random population
        population = [
            ''.join(map(str, random.sample(self.seq, len(self.seq))))
            for _ in range(self.pop_size)
        ]
        epoch = 0
        while True:
            logging.debug(population)
            epoch += 1
            # Evaluate fitness
            weights = list(map(self.fitness, population))
            highest_fitness = max(weights)
            if epoch % 1000 == 0:
                print("Fittest individual in epoch {} has {:0.2f}% fitness".
                      format(epoch, highest_fitness / self.max_fitness * 100))
            if highest_fitness >= self.max_fitness:
                for individual in population:
                    if self.fitness(individual) == highest_fitness:
                        winner = individual
                        break
                print(
                    "Found a sequence {} satisfying {:0.2f}% of `max_fitness`".
                    format(winner, highest_fitness / self.max_fitness * 100))
                return 0

            # Generate new population
            # a) Selection
            # b) Crossover
            # c) Mutation
            new_population = []
            for i in range(0, len(population), 2):
                dad = self.select(population, weights)
                mom = self.select(population, weights)
                son = self.reproduce(dad, mom)
                daughter = self.reproduce(dad, mom)
                if random.random() < self.prob_mutation:
                    son = self.mutate(son)
                if random.random() < self.prob_mutation:
                    daughter = self.mutate(daughter)
                new_population.append(son)
                new_population.append(daughter)

            population = new_population

    def select(self, population, weights):
        cumulative = list(itertools.accumulate(weights))
        x = random.random() * cumulative[-1]
        return population[bisect.bisect(cumulative, x)]

    def reproduce(self, dad, mom):
        crossover_point = random.randint(1, len(dad) - 1)
        return dad[:crossover_point] + mom[crossover_point:]

    def mutate(self, child):
        bit = random.randint(0, len(child) - 1)
        value = random.choice(self.seq)
        logging.debug("VALEU FOR MUTATIONS IS", value)
        return child[:bit] + str(value) + child[bit + 1:]


if __name__ == '__main__':
    logging.basicConfig()
    N = 8
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    nq = NQueen(N)
    GeneticAlgorithm(N, nq.fitness, nq.max_fitness).ga()
