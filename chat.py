from model import UniqueNeuralNetwork
from tokenizer import Tokenizer
import random


def exp_approx(x):
    """Approximate exponential function using Taylor series expansion."""
    if x < -20:
        return 0
    elif x > 20:
        return float('inf')
    result = 1
    term = 1
    for n in range(1, 20):  # Using the first 20 terms for approximation
        term *= x / n
        result += term
    return result

def softmax(output):
    """Calculate softmax probabilities."""
    max_output = max(output)  # Subtract max for numerical stability
    exp_values = [exp_approx(p - max_output) for p in output]
    total = sum(exp_values)
    return [p / total for p in exp_values]

def generate_text(model, tokenizer, seed_text, length=30, temperature=0.4, repetition_penalty=2):
    vocab_size = len(tokenizer.vocab)
    current_tokens = tokenizer.encode(seed_text)
    
    if not current_tokens:
        current_tokens = [tokenizer.vocab['<UNK>']]
    
    generated_text = seed_text
    generated_tokens = []

    for _ in range(length):
        input_vector = [0] * vocab_size
        for token in current_tokens[-5:]:  # Consider last 5 tokens for context
            token_index = min(token, vocab_size - 1)
            input_vector[token_index] = 1

        output = model.forward(input_vector)
        if not output:
            print("Error: Model returned empty output.")
            break

        # Apply temperature scaling
        scaled_output = [exp_approx(p / temperature) for p in output]

        # Apply repetition penalty
        for i in range(len(scaled_output)):
            if i in generated_tokens[-10:]:
                scaled_output[i] /= repetition_penalty

        # Convert to probability distribution using softmax
        probabilities = softmax(scaled_output)

        # Sample from the distribution
        next_token = random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
        next_word = tokenizer.decode([next_token])
        
        generated_text += ' ' + next_word
        current_tokens.append(next_token)
        generated_tokens.append(next_token)

    return generated_text.strip()

def main():
    model_file = 'trained_model.json'
    tokenizer_file = 'trained_tokenizer.json'

    print("Loading model and tokenizer...")
    try:
        model = UniqueNeuralNetwork.load(model_file)
        tokenizer = Tokenizer()
        tokenizer.load(tokenizer_file)
        
        # Ensure model input size matches tokenizer vocabulary size
        if model.input_size != len(tokenizer.vocab):
            print("Warning: Model input size does not match tokenizer vocabulary size.")
            print(f"Model input size: {model.input_size}, Tokenizer vocabulary size: {len(tokenizer.vocab)}")
            print("Reinitializing model...")
            model = UniqueNeuralNetwork(len(tokenizer.vocab), model.hidden_size, len(tokenizer.vocab), model.num_layers)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return

    print("Chat with the AI (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        try:
            ai_response = generate_text(model, tokenizer, user_input, length=30, temperature=0.4, repetition_penalty=2)
            print("AI:", ai_response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("AI: I'm sorry, I encountered an error while generating a response.")

if __name__ == "__main__":
    main()
