# Reference for Learning RAG Pipeline Development

This README is a practical guide for building your intuition and skills in Retrieval-Augmented Generation (RAG). It aims to help you understand the foundational concepts, then connect them to the pieces of a working RAG pipeline.

If you are new to the field, start with the basics:

- Machine learning fundamentals: vectors, matrices, embeddings, and similarity.
- Python fundamentals: data structures, functions, and basic file I/O.
- Deep learning basics: ANNs, activation functions, loss functions, and optimizers.

These topics are not strictly required to run example code, but they help you understand why RAG works and how to improve it.

## 1. Vector Databases, Vector Search, and Embeddings

**Core idea**

- A relational database stores data in rows and columns.
- A vector database stores data as high-dimensional points.

An embedding is a numerical representation of an object (text, image, audio, etc.) that captures meaning. Similar items end up close together in vector space, which makes similarity search possible.

**Simple analogy**

- Relational DB: a 2D table of values.
- Vector DB: a 3D (or higher-dimensional) space of points.

When you store vectors, you are storing points, not rows and columns. For example, an image of a dog can be represented by attributes such as four-legged, mammal, and domestic. Those attributes become dimensions in a vector space.

**Why embeddings matter**

- They let you compare meaning, not just exact words.
- They support similarity search (nearest neighbors).
- They enable retrieval for RAG.
