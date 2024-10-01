# CATF (Custom AI Training Framework)

## Introduction

CATF is a cutting-edge Custom AI Training Framework designed for advanced natural language processing and generation. This project implements a novel neural network architecture with adaptive learning rates and memory cells, offering a fresh approach to language modeling.

## Table of Contents

- [Features](#features)
- [Why Choose CATF](#why-choose-catf)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Detailed Component Overview](#detailed-component-overview)
- [Usage Examples](#usage-examples)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Innovative Neural Network Architecture**
  - Adaptive learning rates for each neuron
  - Integrated memory cells for enhanced context retention
  - Multi-layer design with customizable depth

- **Advanced Tokenization**
  - Custom tokenizer with vocabulary management
  - Support for saving and loading vocabularies
  - Efficient encoding and decoding of text

- **Visual Model Representation**
  - Graph-based visualization using NetworkX
  - Insight into token-neuron relationships

- **Optimized Activation Functions**
  - Custom adaptive activation function
  - Combines benefits of ReLU and sigmoid

- **Efficient Computation**
  - Taylor series expansion for exponential approximation
  - Optimized sigmoid function implementation

- **Flexible Training Process**
  - Customizable epochs, learning rates, and batch sizes
  - Integrated validation set for performance monitoring

## Why Choose CATF

1. **Cutting-edge Architecture**: Explore the potential of adaptive learning rates and memory cells in neural networks.
2. **Transparency**: Gain insights into your model's internal structure through graph-based visualization.
3. **Flexibility**: Easily modify and experiment with various aspects of the model.
4. **Efficiency**: Benefit from optimized implementations of critical functions.
5. **Educational Value**: Ideal for those looking to understand and experiment with custom neural network implementations.

## Getting Started

1. Clone the repository:

2. Install dependencies:

3. Prepare your corpus file (text file containing training data).

4. Run the training script:
```python
from CATF.Progress.trainer import train_model

train_model('path_to_your_corpus.txt', epochs=10, learning_rate=0.01)
```

## Project Structure
``` 
CATF/
├── graph.py
├── model.py
├── chat.py
└── Progress/
    ├── trainer.py
    ├── model.py
    └── tokenizer.py
```

## Detailed Component Overview

## UniqueNeuralNetwork (Progress/model.py)
- The core of CATF, implementing:

    - Multi-layer neural network
    - Adaptive learning rates
    - Memory cells
    - Custom activation functions

## Tokenizer (Progress/tokenizer.py)
- Handles text preprocessing:

    - Vocabulary creation
    - Text encoding and decoding
    - Vocabulary persistence (save/load)

## Graph Creation (graph.py)
- Visualizes the model structure:

    - Creates a directed graph of the neural network
    - Represents tokens and neurons as nodes
    - Edges represent connections with weights

## Training Process (Progress/trainer.py)
- Manages the model training:

    - Corpus preprocessing
    - Batch creation
    - Training loop with validation

## Usage Examples

**Training a Model**
```python
from CATF.Progress.trainer import train_model

train_model('corpus.txt', epochs=20, learning_rate=0.05)
```
**Using the Tokenizer**
```python
from CATF.Progress.tokenizer import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit("Your training text here")
encoded = tokenizer.encode("Text to encode")
decoded = tokenizer.decode(encoded)
```
**Visualizing the Model**
```python
from CATF.graph import create_graph

graph = create_graph(model, tokenizer)
# Use NetworkX to visualize or analyze the graph
```

## Performance Optimization

- CATF includes several optimizations:

    - Custom exponential approximation using Taylor series
    - Safe sigmoid function to handle extreme values
    - Efficient batch processing during training

