import random
import math
import time
import statistics
import matplotlib.pyplot as plt

N = random.randint(1000, 10000)
n = random.randint(3, 10)
start_time = time.time()

cities = []
for i in range(N):
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    cities.append((x, y))

def fitness_function(chromosome, cities, n):
    airports = chromosome_to_airports(chromosome, cities, n)
    nearest_airports = find_nearest_airports(cities, airports)
    distances = [distance_squared(city, nearest_airports[i]) for i, city in enumerate(cities)]
    return sum(distances)

def chromosome_to_airports(chromosome, cities, n):
    airports = []
    for i in range(n):
        airport_index = chromosome[i]
        airport = cities[airport_index]
        airports.append(airport)
    return airports

def find_nearest_airports(cities, airports):
    nearest_airports = []
    for city in cities:
        nearest_airport = min(airports, key=lambda airport: distance(city, airport))
        nearest_airports.append(nearest_airport)
    return nearest_airports

def distance_squared(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

def distance(point1, point2):
    return math.sqrt(distance_squared(point1, point2))


def create_individual(cities, num_airports):
    return random.sample(range(len(cities)), num_airports)

def mutate(chromosome, num_airports):
    index_to_mutate = random.randint(0, num_airports - 1)
    new_value = random.randint(0, len(cities) - 1)
    mutated_chromosome = chromosome[:]
    mutated_chromosome[index_to_mutate] = new_value
    return mutated_chromosome


def genetic_algorithm(cities, num_airports, pop_size, elite_size, mutation_rate, generations):
    population = []
    for i in range(pop_size):
        population.append(create_individual(cities, num_airports))
    
    for generation in range(generations):
        fitness_scores = [(individual, fitness_function(individual, cities, num_airports)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1])
        ranked_individuals = [individual for (individual, fitness_score) in fitness_scores]
        
        elite = ranked_individuals[:elite_size]
            
        mating_pool = []
        for i in range(pop_size - elite_size):
            tournament = random.sample(ranked_individuals, 2)
            winner = max(tournament, key=lambda x: x[1])[0]
            mating_pool.append(winner)
        
        offspring = []
        for i in range(len(mating_pool) // 2):
            parent1 = mating_pool[i*2]
            parent2 = mating_pool[i*2 + 1]
            if not isinstance(parent1, list) or not isinstance(parent2, list):
                continue
best_fitness_values = []
accuracy_list = []
time_list = []
num_iterations = 1000

for i in range(num_iterations):
    cities = []
    N = random.randint(1000, 10000)
    n = random.randint(3, 10)
    for i in range(N):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    start_time = time.time()
    best_individual, best_fitness = genetic_algorithm(cities, n, pop_size=100, elite_size=20, mutation_rate=0.1, generations=100)
    end_time = time.time()
    best_fitness_values.append(best_fitness)
    current_accuracy = 1 / best_fitness
    current_time = end_time - start_time
    accuracy_list.append(current_accuracy)
    time_list.append(current_time)

end_time = time.time()

accuracy = 1 / best_fitness_values[-1]
time_taken = end_time - start_time
mean_accuracy = sum(accuracy_list) / num_iterations
std_accuracy = statistics.stdev(accuracy_list)
mean_time = sum(time_list) / num_iterations
std_time = statistics.stdev(time_list)

print("Genetic Algorithm:")
print("Best solution: ", best_individual)
print("Objective function value: ", best_fitness_values[-1])

print("Mean accuracy: ", mean_accuracy)
print("Standard deviation of accuracy: ", std_accuracy)
print("Mean time: ", mean_time)
print("Standard deviation of time: ", std_time)

print("Accuracy: ", accuracy)
print("Time taken (in seconds): ", time_taken)

plt.plot(best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.title('Convergence Plot for Genetic Algorithm')
plt.show()
