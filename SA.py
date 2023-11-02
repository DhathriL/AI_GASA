import random
import math
import time
import statistics
import matplotlib.pyplot as plt

N = random.randint(1000, 10000)
n = random.randint(3, 10)

cities = []
for i in range(N):
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    cities.append((x, y))

def objective_function(airports, cities):
    nearest_airports = find_nearest_airports(cities, airports)
    distances = [distance(city, nearest_airports[i]) for i, city in enumerate(cities)]
    return sum(distances)

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

def initialize(cities, n):
    return random.sample(range(len(cities)), n)

def perturb(solution, cities, temperature):
    index_to_mutate = random.randint(0, len(solution) - 1)
    new_value = random.randint(0, len(cities) - 1)
    mutated_solution = solution[:]
    mutated_solution[index_to_mutate] = new_value
    return mutated_solution

def accept(current_obj, new_obj, temperature):
    if new_obj < current_obj:
        return True
    delta_obj = new_obj - current_obj
    acceptance_prob = math.exp(-delta_obj / temperature)
    return random.random() < acceptance_prob

def cooling_schedule(temperature, cooling_rate):
    return temperature * cooling_rate

initial_temperature = 1000
cooling_rate = 0.95

num_iterations = 1000

best_obj_values = []
times = []
temperatures = [initial_temperature]

for i in range(num_iterations):
    current_solution = initialize(cities, n)
    current_obj = objective_function(current_solution, cities)
    
    start_time = time.time()

    while initial_temperature > 1:
        new_solution = perturb(current_solution, cities, initial_temperature)
        new_obj = objective_function(new_solution, cities)

        if accept(current_obj, new_obj, initial_temperature):
            current_solution = new_solution
            current_obj = new_obj

        initial_temperature = cooling_schedule(initial_temperature, cooling_rate)
        
        best_obj_values.append(current_obj)
        temperatures.append(initial_temperature)

    end_time = time.time
    
    
    
    current_accuracy = 1 / current_obj
    current_time = end_time - start_time
    
    accuracy_list.append(current_accuracy)
    time_list.append(current_time)
print("Best solution: ", current_solution)
print("Objective function value: ", current_obj)
    
mean_accuracy = sum(accuracy_list) / num_iterations
std_accuracy = statistics.stdev(accuracy_list)
mean_time = sum(time_list) / num_iterations
std_time = statistics.stdev(time_list)

print("Mean accuracy: ", mean_accuracy)
print("Standard deviation of accuracy: ", std_accuracy)
print("Mean time: ", mean_time)
print("Standard deviation of time: ", std_time)

print("Accuracy: ", accuracy)
print("Time taken (in seconds): ", time_taken)

plt.plot(best_obj_values, label='SA')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
