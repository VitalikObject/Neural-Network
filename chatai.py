import random

# Define the activation function for the neurons
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Define the mutation rate and standard deviation for the mutate method
MUTATION_RATE = 0.1
MUTATION_STD = 0.1

class Neuron:
  def __init__(self, weights, bias, activation_function):
    self.weights = weights
    self.bias = bias
    self.activation_function = activation_function
    
  def forward(self, inputs):
    weighted_sum = sum(inputs[i] * self.weights[i] for i in range(len(inputs)))
    return self.activation_function(weighted_sum + self.bias)
  
  def mutate(self):
    # Mutate the weights and bias of the neuron
    for i in range(len(self.weights)):
      if random.random() < MUTATION_RATE:
        self.weights[i] += random.gauss(0, MUTATION_STD)
    if random.random() < MUTATION_RATE:
      self.bias += random.gauss(0, MUTATION_STD)

class ChatNeuralNetwork:
  def __init__(self, input_neurons, hidden_neurons, output_neurons, generation, fitness_score=0.0):
    self.input_neurons = input_neurons
    self.hidden_neurons = hidden_neurons
    self.output_neurons = output_neurons
    self.generation = generation
    self.fitness_score = fitness_score
    
  def forward(self, inputs):
    hidden_outputs = [neuron.forward(inputs) for neuron in self.hidden_neurons]
    return [neuron.forward(hidden_outputs) for neuron in self.output_neurons]
    
  def mutate(self):
    # Mutate the weights and biases of the neurons in the network
    for neuron in self.input_neurons:
      neuron.mutate()
    for neuron in self.hidden_neurons:
      neuron.mutate()
    for neuron in self.output_neurons:
      neuron.mutate()  
    
def create_next_generation(previous_generation):
  offspring = []
  
  # Perform selection and crossover to produce offspring networks
  for i in range(len(previous_generation)):
    parent1 = select_parent(previous_generation)
    parent2 = select_parent(previous_generation)
    child = crossover(parent1, parent2)
    offspring.append(child)
  
  # Mutate the weights and biases of the offspring networks
  for child in offspring:
    child.mutate()
    
  # Return the next generation of chat neural networks
  return [ChatNeuralNetwork(child.input_neurons, child.hidden_neurons, child.output_neurons, generation + 1) for child in offspring]
  
def select_parent(previous_generation):
  # Select a parent based on their fitness score
  fitness_scores = [network.fitness_score for network in previous_generation]
  total_fitness = sum(fitness_scores)
  if total_fitness == 0:
    # If all fitness scores are 0, return a random parent
    return previous_generation[random.randint(0, len(previous_generation) - 1)]
  probs = [score / total_fitness for score in fitness_scores]
  parent_index = np.random.choice(len(previous_generation), p=probs)
  return previous_generation[parent_index]

def crossover(parent1, parent2):
  # Create a child network by combining the weights and biases of the parent networks
  input_neurons = []
  for i in range(len(parent1.input_neurons)):
    weights = []
    for j in range(len(parent1.input_neurons[i].weights)):
      # Choose weights from either parent at random
      weights.append(random.choice([parent1.input_neurons[i].weights[j], parent2.input_neurons[i].weights[j]]))
    bias = random.choice([parent1.input_neurons[i].bias, parent2.input_neurons[i].bias])
    activation_function = parent1.input_neurons[i].activation_function
    input_neurons.append(Neuron(weights, bias, activation_function))
  
  hidden_neurons = []
  for i in range(len(parent1.hidden_neurons)):
    weights = []
    for j in range(len(parent1.hidden_neurons[i].weights)):
      # Choose weights from either parent at random
      weights.append(random.choice([parent1.hidden_neurons[i].weights[j], parent2.hidden_neurons[i].weights[j]]))
    bias = random.choice([parent1.hidden_neurons[i].bias, parent2.hidden_neurons[i].bias])
    activation_function = parent1.hidden_neurons[i].activation_function
    hidden_neurons.append(Neuron(weights, bias, activation_function))
  
  output_neurons = []
  for i in range(len(parent1.output_neurons)):
    weights = []
    for j in range(len(parent1.output_neurons[i].weights)):
      # Choose weights from either parent at random
      weights.append(random.choice([parent1.output_neurons[i].weights[j], parent2.output_neurons[i].weights[j]]))
    bias = random.choice([parent1.output_neurons[i].bias, parent2.output_neurons[i].bias])
    activation_function = parent1.output_neurons[i].activation_function
    output_neurons.append(Neuron(weights, bias, activation_function))
  
  return ChatNeuralNetwork(input_neurons, hidden_neurons, output_neurons, parent1.generation + 1)

  def mutate(self):
    # Mutate the weights and biases of the neurons in the network
    for neuron in self.input_neurons:
      neuron.mutate()
    for neuron in self.hidden_neurons:
      neuron.mutate()
    for neuron in self.output_neurons:
      neuron.mutate()
  
if __name__ == '__main__':
  # Define the activation function for the neurons
  def sigmoid(x):
    return 1 / (1 + math.exp(-x))
  
  # Initialize the input, hidden, and output layers of the chat neural network
  input_layer = [Neuron([0.1, 0.2, 0.3], 0.4, sigmoid) for _ in range(3)]
  hidden_layer = [Neuron([0.5, 0.6, 0.7], 0.8, sigmoid) for _ in range(4)]
  output_layer = [Neuron([0.9, 1.0, 1.1], 1.2, sigmoid) for _ in range(2)]
  
  # Create the initial generation of chat neural networks
  generation = 0
  chat_networks = [ChatNeuralNetwork(input_layer, hidden_layer, output_layer, generation) for _ in range(10)]
  
  # Loop through generations, creating new generations of chat neural networks
  while generation < 10:
    chat_networks = create_next_generation(chat_networks)
    generation += 1
    
    