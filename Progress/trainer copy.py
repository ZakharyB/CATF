"""
import json
import math
import random
import logging
import time
from model import UniqueNeuralNetwork
from tokenizer import Tokenizer

logging.basicConfig(level=logging.DEBUG)

def train_model(corpus_file, epochs, learning_rate):
    logging.info("Starting model training...")

    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus = f.read()
    except FileNotFoundError:
        logging.error(f"Corpus file not found: {corpus_file}")
        raise


    processed_corpus = ""
    for line in corpus.split('\n'):
        if line.startswith('User:') or line.startswith('AI:'):
            processed_corpus += line[line.index(':')+1:].strip() + ' '
        else:
            processed_corpus += line.strip() + ' '

    logging.debug("Corpus preprocessed")

    tokenizer = Tokenizer()
    tokenizer.fit(processed_corpus)


    vocab_size = len(tokenizer.vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    model = UniqueNeuralNetwork(input_size=vocab_size, hidden_size=256, output_size=vocab_size, num_layers=4)
    logging.debug("Model initialized")


    tokens = tokenizer.encode(processed_corpus)
    logging.info(f"Total tokens in corpus: {len(tokens)}")

    # Create a small validation set
    validation_size = min(1000, len(tokens) // 10)
    validation_tokens = tokens[-validation_size:]
    train_tokens = tokens[:-validation_size]

    logging.info(f"Training on {len(train_tokens)} tokens, validating on {len(validation_tokens)} tokens")

    def create_batch(tokens, batch_size):
        for i in range(0, len(tokens) - 1, batch_size):
            yield tokens[i:i+batch_size], tokens[i+1:i+batch_size+1]

    batch_size = 3
    
    start_time = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        logging.debug(f"Starting epoch {epoch+1}")
        for i, (input_batch, target_batch) in enumerate(create_batch(train_tokens, batch_size)):
            batch_loss = 0
            for input_token, target_token in zip(input_batch, target_batch):
                input_vector = [0] * vocab_size
                if input_token < vocab_size:
                    input_vector[input_token] = 1
                target_vector = [0] * vocab_size
                if target_token < vocab_size:
                    target_vector[target_token] = 1
                
                try:
                    logging.debug(f"Processing token pair: {input_token}, {target_token}")
                    output = model.forward(input_vector)
                    loss = -sum(t * math.log(o + 1e-10) for t, o in zip(target_vector, output))
                    batch_loss += loss
                    
                    model.train(input_vector, target_vector, learning_rate)
                except Exception as e:
                    logging.error(f"Error during training: {e}")
                    logging.error(f"Input token: {input_token}, Target token: {target_token}")
                    raise
            
            total_loss += batch_loss / batch_size
            
            if i % 10 == 0:  # Print progress every 10 batches
                logging.info(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_tokens)//batch_size}, Loss: {batch_loss/batch_size:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f} seconds, Average Loss: {total_loss/(len(train_tokens)//batch_size):.4f}")

        # Validation
        val_loss = 0
        for input_token, target_token in zip(validation_tokens[:-1], validation_tokens[1:]):
            input_vector = [0] * vocab_size
            if input_token < vocab_size:
                input_vector[input_token] = 1
            target_vector = [0] * vocab_size
            if target_token < vocab_size:
                target_vector[target_token] = 1
            
            output = model.forward(input_vector)
            loss = -sum(t * math.log(o + 1e-10) for t, o in zip(target_vector, output))
            val_loss += loss

        val_loss /= len(validation_tokens) - 1

        logging.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss/len(train_tokens):.4f}, Validation Loss: {val_loss:.4f}")

        # Generate a sample text after each epoch
        sample_input = random.choice(train_tokens)
        sample_text = tokenizer.decode([sample_input])
        generated_text = generate_sample(model, tokenizer, sample_text, length=20)
        logging.info(f"Sample generated text: {generated_text}")

        # Check if training is taking too long
        if time.time() - start_time > 10:  # 1 hour
            logging.warning("Training has been running for over an hour. Stopping early.")
            break

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")

    return model, tokenizer

def generate_sample(model, tokenizer, seed_text, length=20):
    current_tokens = tokenizer.encode(seed_text)
    generated_text = seed_text

    for _ in range(length):
        input_vector = [0] * len(tokenizer.vocab)
        for token in current_tokens[-5:]:  # Consider last 5 tokens for context
            if token < len(tokenizer.vocab):
                input_vector[token] = 1
        output = model.forward(input_vector)
        next_token = random.choices(range(len(output)), weights=output, k=1)[0]
        next_word = tokenizer.decode([next_token])
        generated_text += ' ' + next_word
        current_tokens.append(next_token)

    return generated_text

if __name__ == "__main__":
    try:
        model, tokenizer = train_model('data/corpus.txt', epochs=10, learning_rate=0.01)
        model.save('trained_model.json')
        tokenizer.save('trained_tokenizer.json')
        logging.info("Training completed. Model and tokenizer saved.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise
        
        """