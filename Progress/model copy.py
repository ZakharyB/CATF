"""
import math
import random
import json
import logging

class UniqueNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, num_layers=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.learned_data = {}
        
        # Initialize layers
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                layer_input_size = input_size
            else:
                layer_input_size = hidden_size
            
            layer_output_size = hidden_size if i < num_layers - 1 else output_size
            self.layers.append(self.initialize_layer(layer_input_size, layer_output_size))
        
        # Unique feature: Adaptive learning rates for each neuron
        self.learning_rates = [[random.uniform(0.01, 0.1) for _ in range(layer_output_size)] for layer in self.layers]
        
        # Unique feature: Memory cells
        self.memory = [0] * hidden_size

    def initialize_layer(self, input_size, output_size):
        return {
            'weights': [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(output_size)],
            'bias': [random.uniform(-0.1, 0.1) for _ in range(output_size)]
        }

    def adaptive_activation(self, x):
        return max(0, min(x, 10)) * self.safe_sigmoid(x)

    def safe_sigmoid(self, x):
        if x < -709:
            return 0
        elif x > 709:
            return 1
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        logging.debug(f"Forward pass input: {inputs[:5]}...")
        current_input = inputs
        for i, layer in enumerate(self.layers):
            output = []
            for j, (weights, bias) in enumerate(zip(layer['weights'], layer['bias'])):
                neuron_output = sum(i * w for i, w in zip(current_input, weights)) + bias
                if i < len(self.layers) - 1:  # Apply adaptive activation to hidden layers
                    neuron_output = self.adaptive_activation(neuron_output)
                output.append(neuron_output)
            
            if i == 0:  # After first hidden layer, update memory
                self.update_memory(output)
            
            current_input = output + self.memory  # Add memory to input of next layer

            logging.debug(f"Layer {i+1} output: {current_input[:5]}...")
        
        # Apply softmax to the final output
        exp_output = [math.exp(x) for x in current_input[:self.output_size]]
        sum_exp_output = sum(exp_output)
        return [x / sum_exp_output for x in exp_output]
    
    def update_memory(self, hidden_output):
        # Update memory cells based on current hidden layer output
        for i in range(len(self.memory)):
            self.memory[i] = 0.9 * self.memory[i] + 0.1 * hidden_output[i]

    def train(self, inputs, targets, global_learning_rate):
        # Forward pass
        layer_inputs = [inputs]
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            outputs = []
            for weights, bias in zip(layer['weights'], layer['bias']):
                neuron_output = sum(i * w for i, w in zip(layer_inputs[-1], weights)) + bias
                if i < len(self.layers) - 1:
                    neuron_output = self.adaptive_activation(neuron_output)
                else:
                    neuron_output = self.safe_sigmoid(neuron_output)
                outputs.append(neuron_output)
            layer_outputs.append(outputs)
            layer_inputs.append(outputs + self.memory)

        # Backward pass
        deltas = []
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                # Output layer
                layer_delta = [
                    output * (1 - output) * (target - output)
                    for output, target in zip(layer_outputs[i], targets)
                ]
            else:
                # Hidden layers
                next_weights = self.layers[i+1]['weights']
                next_delta = deltas[-1]
                layer_delta = [
                    self.adaptive_activation(output) * sum(nd * nw[j] for nd, nw in zip(next_delta, next_weights))
                    for j, output in enumerate(layer_outputs[i])
                ]
            deltas.append(layer_delta)

        # Update weights and biases
        deltas.reverse()
        for i, layer in enumerate(self.layers):
            for j, (weights, bias) in enumerate(zip(layer['weights'], layer['bias'])):
                for k in range(len(weights)):
                    adaptive_lr = self.learning_rates[i][j] * global_learning_rate
                    gradient = deltas[i][j] * layer_inputs[i][k]
                    # Gradient clipping
                    gradient = max(min(gradient, 1), -1)
                    weights[k] += adaptive_lr * gradient
                    self.learning_rates[i][j] *= 1.01 if random.random() > 0.5 else 0.99
                bias += global_learning_rate * deltas[i][j]
                # Clip bias
                bias = max(min(bias, 1), -1)

        # Update memory
        self.update_memory(layer_outputs[0])

    def learn(self, input_text, response):
        # Simple learning: associate input with response
        if input_text not in self.learned_data:
            self.learned_data[input_text] = []
        self.learned_data[input_text].append(response)

    def get_learned_response(self, input_text):
        if input_text in self.learned_data:
            return random.choice(self.learned_data[input_text])
        return None

    def save(self, filename):
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'layers': self.layers,
            'learned_data': self.learned_data,
            'learning_rates': self.learning_rates,
            'memory': self.memory
        }
        
        # Convert numpy arrays to lists if you're using numpy
        def convert_to_serializable(obj):
            if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
                return obj
            return str(obj)
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, default=convert_to_serializable, indent=2)
        
        print(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            model_data = json.load(f)
        
        # Create a new instance of the model
        model = cls(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size'],
            num_layers=model_data['num_layers']
        )
        
        # Load the layers
        model.layers = model_data['layers']
        
        # Load the learning rates
        model.learning_rates = model_data['learning_rates']
        
        # Load the memory
        model.memory = model_data['memory']
        
        print(f"Model loaded from {filename}")
        model.learned_data = model_data.get('learned_data', {})

        return model

    # Helper method to initialize a layer (moved from __init__)
    def initialize_layer(self, input_size, output_size):
        return {
            'weights': [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(output_size)],
            'bias': [random.uniform(-1, 1) for _ in range(output_size)]
        }

        """