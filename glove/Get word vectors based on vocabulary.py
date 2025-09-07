import pickle
import torch
import numpy as np
import os
import random

def load_glove_vectors(file_path, word_index):
    """
    Load GloVe vectors and build embedding matrix

    Parameters:
        file_path: GloVe file path
        word_index: word to index mapping dictionary

    Returns:
        Embedding matrix (numpy array)
    """
    # Initialize embedding matrix
    embedding_matrix = np.zeros((len(word_index), 300), dtype=np.float32)
    found_words = set()

    # Read GloVe file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            if len(values) < 301:  # Ensure there are enough values
                continue

            word = values[0]
            # If word is in vocabulary
            if word in word_index:
                index = word_index[word]
                vector = np.asarray(values[1:301], dtype=np.float32)
                embedding_matrix[index] = vector
                found_words.add(word)

    print(f"Number of words found in GloVe: {len(found_words)}/{len(word_index)}")
    return embedding_matrix, found_words


def build_vocab_index(vocab_file):
    """
    Build word index from vocabulary file

    Parameters:
        vocab_file: vocabulary file path (one word per line)

    Returns:
        word_index: word to index mapping dictionary
        index_word: index to word mapping dictionary
    """
    # Read vocabulary file
    with open(vocab_file, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]

    # Create word index mapping
    word_index = {}
    index_word = {}

    # Add special tokens
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    # Add words from vocabulary
    for idx, word in enumerate(words):
        word_index[word] = idx + 2  # Start indexing from 2

    # Create reverse mapping
    for word, idx in word_index.items():
        index_word[idx] = word

    return word_index, index_word, words

def generate_random_vector(mean=0.0, std=0.1):
    """Generate random vector, simulating GloVe distribution characteristics"""
    return np.random.normal(mean, std, 300).astype(np.float32)

name = 'TREC'  # Custom name    ohsumed    TagMyNews  StackOverflow    mr_vocabulary_max_new   snippets    Twitter   mr   R52
# Configuration parameters
vocab_file = 'vocabulary_TREC.txt'  # Vocabulary file    StackOverflow_vocabulary_max_new   vocabulary_snippets
# vocab_file = 'vocabulary.txt'  # Vocabulary file    StackOverflow_vocabulary_max_new  Twitter
glove_file_path = "glove.6B.300d.txt"  # GloVe file path
# name = 'ohsumed'  # Custom name    ohsumed    TagMyNews  StackOverflow

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Step 1: Build word index from vocabulary file
word_index, index_word, vocab_words = build_vocab_index(vocab_file)
print(f"Vocabulary size: {len(word_index)} (includes 2 special tokens)")

# Step 2: Load GloVe vectors
embedding_matrix, found_words = load_glove_vectors(glove_file_path, word_index)

# Analyze statistical characteristics of GloVe vectors for generating random vectors
# Calculate mean and standard deviation of found word vectors
if len(found_words) > 0:
    found_vectors = embedding_matrix[[word_index[w] for w in found_words]]
    mean_val = np.mean(found_vectors)
    std_val = np.std(found_vectors)
    print(f"Found word vector statistics: mean={mean_val:.6f}, std={std_val:.6f}")
else:
    mean_val = 0.0
    std_val = 0.1  # Default value

# Handle not found words - use random vectors
not_found_count = 0
for word in word_index:
    if word not in found_words and word not in ["<PAD>", "<UNK>"]:
        idx = word_index[word]
        # Generate random vector for each not found word
        embedding_matrix[idx] = generate_random_vector(mean_val, std_val)
        not_found_count += 1

print(f"Generated random vectors for {not_found_count} not found words")

# Create vectors for special tokens
pad_emb_npa = np.zeros((1, 300))  # <PAD> vector
unk_emb_npa = generate_random_vector(mean_val, std_val).reshape(1, -1)  # <UNK> vector

# Combine embedding matrix
embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embedding_matrix[2:]))

# Update word index to match new matrix
new_word_index = {}
new_word_index["<PAD>"] = 0
new_word_index["<UNK>"] = 1

# Remap original vocabulary index
for idx, word in enumerate(vocab_words):
    new_word_index[word] = idx + 2

# Convert to PyTorch tensor
embedding_tensor = torch.from_numpy(embs_npa)
print(f"Embedding tensor shape: {embedding_tensor.shape}")

# Create output directory
os.makedirs('../glove', exist_ok=True)

# Save embedding tensor
torch.save(embedding_tensor, f'../glove/embedding_{name}.pt')