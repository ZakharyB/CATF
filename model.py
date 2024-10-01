import random
import json
import logging
import os

class UniqueNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 5, 
                 settings: dict = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.settings = settings or {
            'repetition_penalty': 1.2,
            'temperature': 0.8,
            'top_k': 40,
            'top_p': 0.9,
            'max_length': 100,
        }

        self.layers = [self.initialize_layer(input_size if i == 0 else hidden_size, 
                                              hidden_size if i < num_layers - 1 else output_size)
                       for i in range(num_layers)]
        
        self.memory = [0] * hidden_size


    def to_dict(self):
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'layers': self.layers,
            'memory': self.memory,
            'settings': self.settings
        }
    
    def initialize_layer(self, input_size: int, output_size: int) -> dict:
        return {
            'weights': [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(output_size)],
            'bias': [random.uniform(-0.1, 0.1) for _ in range(output_size)]
        }

    def adaptive_activation(self, x: float) -> float:
        return max(0, min(x, 5)) * self.safe_sigmoid(x)

    def safe_sigmoid(self, x: float) -> float:
        """Approximate sigmoid function."""
        if x < -709:
            return 0
        elif x > 709:
            return 1
        return 1 / (1 + 1 / (2.718281828459045 ** x))

    def forward(self, inputs: list) -> list:
        current_input = inputs
        for i, layer in enumerate(self.layers):
            neuron_output = []
            for j, (weights, bias) in enumerate(zip(layer['weights'], layer['bias'])):
                output = sum(i * w for i, w in zip(current_input, weights)) + bias
                neuron_output.append(self.adaptive_activation(output))
            
            if i == 0:
                self.update_memory(neuron_output)

            current_input = neuron_output + self.memory

        exp_output = self.safe_exp(current_input[:self.output_size])
        sum_exp_output = sum(exp_output)
        return [x / sum_exp_output for x in exp_output]

    def safe_exp(self, x: list) -> list:
        """Compute e^x for a list of values with clipping."""
        return [self.exp_clipped(v) for v in x]

    def exp_clipped(self, x: float) -> float:
        """Clipped exponential function."""
        if x < -709:
            return 0
        elif x > 709:
            return float('inf')
        return 2.718281828459045 ** x

    def update_memory(self, hidden_output: list):
        self.memory = [0.9 * m + 0.1 * h for m, h in zip(self.memory, hidden_output)]

    def train(self, inputs: list, targets: list, learning_rate: float):
        layer_inputs = [inputs]
        layer_outputs = []

        for layer in self.layers:
            outputs = []
            for weights, bias in zip(layer['weights'], layer['bias']):
                neuron_output = sum(i * w for i, w in zip(layer_inputs[-1], weights)) + bias
                outputs.append(self.adaptive_activation(neuron_output))
            layer_outputs.append(outputs)
            layer_inputs.append(outputs + self.memory)

        deltas = self.calculate_deltas(layer_outputs, targets)

        for i, layer in enumerate(self.layers):
            self.update_weights(layer, layer_inputs[i], deltas[i], learning_rate)

        self.update_memory(layer_outputs[0])

    def calculate_deltas(self, layer_outputs: list, targets: list) -> list:
        deltas = []
        for i in reversed(range(len(self.layers))):
            layer_output = layer_outputs[i]
            if i == len(self.layers) - 1:
                layer_delta = [
                    output * (1 - output) * (target - output)
                    for output, target in zip(layer_output, targets)
                ]
            else:
                next_weights = self.layers[i + 1]['weights']
                next_delta = deltas[-1]
                layer_delta = [
                    self.adaptive_activation(output) * sum(nd * nw[j] for nd, nw in zip(next_delta, next_weights))
                    for j, output in enumerate(layer_output)
                ]
            deltas.append(layer_delta)
        return deltas[::-1]

    def update_weights(self, layer: dict, inputs: list, deltas: list, learning_rate: float):
        for j in range(len(layer['weights'])):
            for k in range(len(layer['weights'][j])):
                gradient = deltas[j] * inputs[k]
                layer['weights'][j][k] += learning_rate * gradient
            layer['bias'][j] += learning_rate * deltas[j]

    def generate_text(self, tokenizer, seed_text: str, max_length: int = None) -> str:
        if max_length is None:
            max_length = self.settings['max_length']

        tokens = tokenizer.encode(seed_text)
        generated = tokens.copy()

        for _ in range(max_length):
            input_vector = self.create_input_vector(generated, tokenizer)
            output = self.forward(input_vector)

            next_token = self.sample_token(output, generated)
            generated.append(next_token)

            if next_token == tokenizer.vocab.get('<EOS>', len(tokenizer.vocab)):
                break

        return tokenizer.decode(generated)

    def sample_token(self, output: list, generated: list) -> int:
        # Apply repetition penalty
        for i in range(len(output)):
            if i in generated[-10:]:
                output[i] /= self.settings['repetition_penalty']

        # Apply temperature
        output = [self.exp_scaled(p) for p in output]

        # Top-K sampling
        top_k_indices = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:self.settings['top_k']]
        top_k_probs = [output[i] for i in top_k_indices]

        # Top-P (nucleus) sampling
        cumulative_probs = [sum(top_k_probs[:i + 1]) for i in range(len(top_k_probs))]
        
        try:
            top_p_index = next(i for i, cp in enumerate(cumulative_probs) if cp > self.settings['top_p'])
        except StopIteration:
            top_p_index = len(cumulative_probs) - 1

        top_p_probs = top_k_probs[:top_p_index + 1]
        top_p_indices = top_k_indices[:top_p_index + 1]

        if not top_p_indices:
            logging.warning("No tokens to choose from. Ending generation.")
            return generated[-1]

        return random.choices(top_p_indices, weights=top_p_probs, k=1)[0]

    def exp_scaled(self, p: float) -> float:
        """Compute e^(p / temperature) without using math.exp."""
        return self.exp_clipped(p / self.settings['temperature'])

    def create_input_vector(self, tokens: list, tokenizer) -> list:
        vector = [0] * self.input_size
        for token in tokens[-5:]:  # Consider last 5 tokens for context
            if token < self.input_size:
                vector[token] = 1
        return vector

    def save(self, filename: str):
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'num_layers': self.num_layers,
            'layers': self.layers,
            'memory': self.memory,
            'settings': self.settings
        }

        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)

        logging.info(f"Model saved to {filename}")

    @classmethod
    def load(cls, filename: str):
        with open(filename, 'r') as f:
            model_data = json.load(f)

        model = cls(
            input_size=model_data['input_size'],
            hidden_size=model_data['hidden_size'],
            output_size=model_data['output_size'],
            num_layers=model_data['num_layers'],
            settings=model_data.get('settings')
        )

        model.layers = model_data['layers']
        model.memory = model_data['memory']

        logging.info(f"Model loaded from {filename}")
        return model

    def save_checkpoint(self, epoch: int, checkpoint_dir: str):
        """Save a checkpoint of the model."""
        checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.json')
        self.save(checkpoint_filename)
        logging.info(f"Checkpoint saved to {checkpoint_filename}")
