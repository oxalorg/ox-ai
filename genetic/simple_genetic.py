"""
The simplest program I can come up with based on the
limited knowledge I have about genetic algorithms
"""
import logging
import bisect
import random
import itertools

class NQueen():
    def __init__(self):
        self.size = 8
        self.max_fitness = 28

    def fitness(self, sol):
        """sol is a list with row number of ith column eg. [2341678]"""
        if len(sol) != self.size:
            print("ERROR IN")
            print(sol)
            raise Exception
        sol = [int(i) for i in sol]
        a_pairs = 0 # attacking pairs
        for queen in range(self.size):
            for ahead_queen in range(queen+1, self.size):
                # each queen ahead of current queen
                if sol[queen] == sol[ahead_queen]:
                    # same row
                    a_pairs += 1
                if abs(sol[ahead_queen] - sol[queen]) == ahead_queen - queen:
                    # diagonals
                    a_pairs += 1

        na_pairs = 7+6+5+4+3+2+1 - a_pairs
        return na_pairs


class GeneticAlgorithm():
    def __init__(self):
        self.pop_size = 10
        self.NQueen = NQueen()
        self.fitness = self.NQueen.fitness
        self.max_fitness = self.NQueen.max_fitness
        self.seq = [1,2,3,4,5,6,7,8]

    def ga(self):
        """Genetic algorithm"""
        # Generate an initial random population
        # population = [''.join(random.choice(self.seq) for _ in range(len(self.seq)) for _ in range(self.pop_size))]
        population = [''.join(map(str, random.sample(self.seq, len(self.seq)))) for _ in range(self.pop_size)]
        epoch = 0
        while True:
            logging.debug(population)
            epoch += 1
            # Evaluate fitness
            weights = list(map(self.fitness, population))
            highest_fitness = max(weights)
            print("Fittest individual in epoch {} has {} fitness".format(epoch, highest_fitness))
            if highest_fitness >= self.max_fitness:
                return population[0]

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
                if random.random() < 0.05:
                    son = self.mutate(son)
                if random.random() < 0.05:
                    daughter = self.mutate(daughter)
                new_population.append(son)
                new_population.append(daughter)

            population = new_population

    def select(self, population, weights):
        cumulative = list(itertools.accumulate(weights))
        x = random.random() * cumulative[-1]
        return population[bisect.bisect(cumulative, x)]


    def reproduce(self, dad, mom):
        crossover_point = random.randint(1, len(dad)-1)
        return dad[:crossover_point] + mom[crossover_point:]

    def mutate(self, child):
        bit = random.randint(0, len(child)-1)
        value = random.choice(self.seq)
        logging.debug("VALEU FOR MUTATIONS IS", value)
        return child[:bit] + str(value) + child[bit+1:]

if __name__ == '__main__':
    logging.basicConfig()
    GeneticAlgorithm().ga()
    # a = NQueen().fitness('24748552')
    # print(a)
