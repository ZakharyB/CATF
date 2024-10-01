import logging
import time
import random
import math
import json
import asyncio

# Assuming UniqueNeuralNetwork and Tokenizer are defined elsewhere
from model import UniqueNeuralNetwork
from tokenizer import Tokenizer

logging.basicConfig(level=logging.DEBUG)

def load_corpus(corpus_file: str) -> str:
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            return f.read().lower()
    except FileNotFoundError:
        logging.error(f"Corpus file not found: {corpus_file}")
        raise

def split_data(tokens: list, split_ratio: float = 0.8) -> tuple:
    split_index = int(len(tokens) * split_ratio)
    return tokens[:split_index], tokens[split_index:]

def augment_data(tokens: list, augment_factor: int = 2) -> list:
    augmented_data = tokens.copy()
    for _ in range(augment_factor):
        random.shuffle(augmented_data)
        augmented_data.extend(tokens)
    return augmented_data

def calculate_metrics(predictions: list, targets: list) -> dict:
    tp = sum(p == t == 1 for p, t in zip(predictions, targets))
    fp = sum(p == 1 and t == 0 for p, t in zip(predictions, targets))
    fn = sum(p == 0 and t == 1 for p, t in zip(predictions, targets))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def save_model(model, filename: str):
    with open(filename, 'w') as f:
        json.dump(model.to_dict(), f)

def load_model(filename: str):
    with open(filename, 'r') as f:
        model_data = json.load(f)
    return UniqueNeuralNetwork.from_dict(model_data)

async def train_epoch(model, train_tokens, tokenizer, vocab_size, learning_rate):
    total_loss = 0
    start_time = time.time()

    for i in range(len(train_tokens) - 1):
        input_vector = model.create_input_vector(train_tokens[:i + 1], tokenizer)
        target_vector = [0] * vocab_size
        if train_tokens[i + 1] < vocab_size:
            target_vector[train_tokens[i + 1]] = 1

        output = model.forward(input_vector)
        loss = -sum(t * math.log(o + 1e-10) for t, o in zip(target_vector, output))
        total_loss += loss
        model.train(input_vector, target_vector, learning_rate)

        if i % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_token = elapsed_time / (i + 1)
            remaining_tokens = len(train_tokens) - 1 - i
            estimated_time_left = avg_time_per_token * remaining_tokens
            logging.debug(
                f"Processed token {i + 1}/{len(train_tokens)}, "
                f"Current loss: {loss:.4f}, "
                f"Estimated time left: {estimated_time_left:.2f} seconds | "
                f"AVG time per token: {avg_time_per_token:.2f} seconds"
            )

    return total_loss / len(train_tokens)

async def validate_model(model, validation_tokens, tokenizer, vocab_size) -> float:
    val_loss = 0
    predictions = []
    targets = []
    
    for i in range(len(validation_tokens) - 1):
        input_vector = model.create_input_vector(validation_tokens[:i + 1], tokenizer)
        target_vector = [0] * vocab_size
        if validation_tokens[i + 1] < vocab_size:
            target_vector[validation_tokens[i + 1]] = 1
        
        output = model.forward(input_vector)
        predictions.append(output.index(max(output)))
        targets.append(validation_tokens[i + 1])

        loss = -sum(t * math.log(o + 1e-10) for t, o in zip(target_vector, output))
        val_loss += loss

    metrics = calculate_metrics(predictions, targets)
    logging.info(f"Validation Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}")

    return val_loss / (len(validation_tokens) - 1)

async def train_model(corpus_file, epochs, learning_rate, patience=5):
    logging.info("Starting model training...")
    corpus = load_corpus(corpus_file)

    tokenizer = Tokenizer()
    tokenizer.fit(corpus)

    vocab_size = len(tokenizer.vocab)
    logging.info(f"Vocabulary size: {vocab_size}")

    model = UniqueNeuralNetwork(input_size=vocab_size, hidden_size=10, output_size=vocab_size, num_layers=3)
    logging.debug("Model initialized")

    tokens = tokenizer.encode(corpus)
    augmented_tokens = augment_data(tokens)
    logging.info(f"Total tokens after augmentation: {len(augmented_tokens)}")

    train_tokens, validation_tokens = split_data(augmented_tokens)
    logging.info(f"Training on {len(train_tokens)} tokens, Validation on {len(validation_tokens)} tokens")

    start_time = time.time()
    prev_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(0, epochs, 2):
        epoch_tasks = []
        for e in range(2):
            if epoch + e < epochs:  # Check for remaining epochs
                epoch_tasks.append(train_epoch(model, train_tokens, tokenizer, vocab_size, learning_rate))
        
        avg_losses = await asyncio.gather(*epoch_tasks)
        
        for e in range(len(avg_losses)):
            avg_loss = avg_losses[e]
            logging.info(f"Epoch {epoch + e + 1}/{epochs} completed, Average Loss: {avg_loss:.4f}")

            # Validation
            avg_val_loss = await validate_model(model, validation_tokens, tokenizer, vocab_size)
            logging.info(f"Validation Loss: {avg_val_loss:.4f}")

            # Check for early stopping
            if avg_val_loss > prev_val_loss:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info("Validation loss increased. Stopping early.")
                    return model, tokenizer
            else:
                patience_counter = 0
                prev_val_loss = avg_val_loss

            # Checkpointing
            if (epoch + e + 1) % 2 == 0:
                checkpoint_filename = f"model_checkpoint_epoch_{epoch + e + 1}.json"
                save_model(model, checkpoint_filename)
                logging.info(f"Model checkpoint saved: {checkpoint_filename}")

            # Learning Rate Scheduler
            if avg_val_loss < prev_val_loss:
                learning_rate *= 0.95

    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")

    return model, tokenizer

def generate_sample(model, tokenizer, seed_text, length=20):
    return model.generate_text(tokenizer, seed_text, max_length=length)

if __name__ == "__main__":
    try:
        model, tokenizer = asyncio.run(train_model('data/corpus.txt', epochs=5, learning_rate=0.001, patience=5))
        model.save('trained_model.json')
        tokenizer.save('trained_tokenizer.json')
        logging.info("Training completed. Model and tokenizer saved.")

        # Generate some sample text
        seed_texts = ["Hello, how are", "The weather is", "I love to"]
        for seed in seed_texts:
            generated = generate_sample(model, tokenizer, seed, length=30)
            print(f"Seed: '{seed}'\nGenerated: '{generated}'\n")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
