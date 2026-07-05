"""
fractal_brain/rag.py
In‑memory vector index, retrieval, and state‑RAG fusion with cross‑attention.
Pure Python, uses math_utils only.
"""
import math
from .math_utils import Vector, Matrix, softmax

class VectorStore:
    """
    A simple brute‑force vector database using dot‑product similarity.
    """
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []   # list of Vector
        self.doc_ids = []   # list of document identifiers (int or str)

    def add(self, vector, doc_id):
        """Add a vector with an associated document id."""
        if not isinstance(vector, Vector):
            vector = Vector(vector)
        assert len(vector) == self.dim
        self.vectors.append(vector)
        self.doc_ids.append(doc_id)

    def search(self, query, k=5):
        """
        Return the top‑k document ids and their similarity scores.
        query: Vector (or list) of length dim.
        """
        if not isinstance(query, Vector):
            query = Vector(query)
        scores = []
        for i, vec in enumerate(self.vectors):
            # dot product similarity
            sim = query.dot(vec)
            scores.append((sim, i))
        # sort descending by similarity
        scores.sort(reverse=True, key=lambda x: x[0])
        top_k = scores[:k]
        doc_ids = [self.doc_ids[i] for _, i in top_k]
        sims = [s for s, _ in top_k]
        return doc_ids, sims

    def get_vector(self, doc_id):
        """Retrieve the stored vector for a document id."""
        # find index by doc_id
        for i, did in enumerate(self.doc_ids):
            if did == doc_id:
                return self.vectors[i]
        return None


class StateRAGFusion:
    """
    Fuses a state embedding with retrieved document embeddings using
    scaled dot‑product cross‑attention (query attends to documents).
    """
    def __init__(self, d_model):
        self.d_model = d_model
        # learnable projections
        self.W_q = Matrix.he_init(d_model, d_model)   # query projection
        self.W_k = Matrix.he_init(d_model, d_model)   # key projection
        self.W_v = Matrix.he_init(d_model, d_model)   # value projection
        self.W_o = Matrix.he_init(d_model, d_model)   # output projection

    def forward(self, state_emb, retrieved_embs):
        """
        state_emb: list or Vector of length d_model
        retrieved_embs: list of lists (or Vectors), each length d_model
        Returns: fused vector (list of length d_model)
        """
        if not isinstance(state_emb, Vector):
            state_emb = Vector(state_emb)
        num_docs = len(retrieved_embs)

        # Convert retrieved embs to list of Vectors
        docs = []
        for e in retrieved_embs:
            if not isinstance(e, Vector):
                docs.append(Vector(e))
            else:
                docs.append(e)

        # Project state (query) into Q
        # state_emb is (d_model,) so treat as row vector; Q = state_emb @ W_q -> Vector
        Q = self.W_q.linear(state_emb)  # Vector (d_model,)

        # Project each document into K and V
        K = Matrix([self.W_k.linear(doc).to_list() for doc in docs])  # (num_docs, d_model)
        V = Matrix([self.W_v.linear(doc).to_list() for doc in docs])  # (num_docs, d_model)

        # Scaled dot‑product attention: scores = Q @ K^T / sqrt(d_model)
        # Q is Vector, K is Matrix; compute dot product between Q and each row of K
        scores = []
        for i in range(num_docs):
            k_vec = Vector(K.data[i])
            dot = Q.dot(k_vec)
            scores.append(dot / math.sqrt(self.d_model))
        # softmax over scores
        attn_weights = softmax(Vector(scores))  # Vector length num_docs

        # Weighted sum of V
        fused = Vector.zeros(self.d_model)
        for i in range(num_docs):
            weight = attn_weights[i]
            v_vec = Vector(V.data[i])
            fused = Vector([fused[j] + weight * v_vec[j] for j in range(self.d_model)])

        # Output projection
        out = self.W_o.linear(fused)  # Vector (d_model,)
        return out.to_list()