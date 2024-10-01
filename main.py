from model import UniqueNeuralNetwork
from tokenizer import Tokenizer
from trainer import train_model
import json
import random
import os

def generate_text(model, tokenizer, seed_text, length=50):
    current_tokens = tokenizer.encode(seed_text)
    generated_text = seed_text

    for _ in range(length):
        input_vector = [0] * len(tokenizer.vocab)
        for token in current_tokens[-5:]:  # Consider last 5 tokens for context
            input_vector[token] = 1
        output = model.forward(input_vector)
        next_token = random.choices(range(len(output)), weights=output, k=1)[0]
        next_word = tokenizer.decode([next_token])
        generated_text += ' ' + next_word
        current_tokens.append(next_token)

    return generated_text

def main():
    model_file = 'trained_model.json'
    tokenizer_file = 'trained_tokenizer.json'

    if os.path.exists(model_file) and os.path.exists(tokenizer_file):
        print("Loading existing model and tokenizer...")
        model = UniqueNeuralNetwork.load(model_file)
        tokenizer = Tokenizer()
        tokenizer.load(tokenizer_file)
    else:
        print("Training new model...")
        model, tokenizer = train_model('data/corpus.txt', epochs=15, learning_rate=0.01)
        model.save(model_file)
        tokenizer.save(tokenizer_file)

    print("Chat with the AI (type 'quit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        ai_response = generate_text(model, tokenizer, user_input)
        print("AI:", ai_response)
        
        # Learning step
        learn = input("Should I learn this response? (y/n): ")
        if learn.lower() == 'y':
            model.learn(user_input, ai_response)
            print("Response learned!")

    print("Saving model and learned data...")
    model.save(model_file)
    with open('learned_data.json', 'w') as f:
        json.dump(model.learned_data, f)
    print("Model and learned data saved. Goodbye!")

if __name__ == "__main__":
    main()