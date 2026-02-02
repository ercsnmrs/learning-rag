import numpy as np
import matplotlib.pyplot as plt

# Sample embeddings for a few words
word_embeddings = {
    "cat": [-0.85, 0.55],
    "dog": [-0.80, 0.52],
    "human": [0.99, 0.90],
    "monkey": [0.75, 0.83],
}

# Plotting for Visualization
plt.figure(figsize=(10, 8))
for word, embedding in word_embeddings.items():
    plt.scatter(embedding[0], embedding[1], label=word)
    plt.annotate(word, (embedding[0], embedding[1]))

plt.title("Simple 2D Word Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True)
plt.show()

# Print Embeddings or Plot
for word, embedding in word_embeddings.items():
    print(f"{word}: {embedding}")