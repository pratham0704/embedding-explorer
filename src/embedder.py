from sentence_transformers import SentenceTransformer
import numpy as np
import os

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts, batch_size=64, show_progress=True):
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        if len(texts) == 0:
            raise ValueError("texts list cannot be empty")

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("every item in texts must be a string")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings
    

    def embed_tickets(self, tickets, batch_size=64):
        if not isinstance(tickets, list):
            raise TypeError("tickets must be a list of ticket dictionaries")

        if len(tickets) == 0:
            raise ValueError("tickets list cannot be empty")

        texts = []

        for ticket in tickets:
            if not isinstance(ticket, dict):
                raise TypeError("each ticket must be a dictionary")

            if "text" not in ticket:
                raise KeyError("each ticket must contain a 'text' field")

            if not isinstance(ticket["text"], str):
                raise TypeError("ticket['text'] must be a string")

            texts.append(ticket["text"])

        print(f"Embedding {len(tickets)} tickets with {self.model_name}...")

        return self.embed(
            texts=texts,
            batch_size=batch_size,
            show_progress=True,
        )
    
    def save_embeddings(self, embeddings, path):
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings must be a numpy array")

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        np.save(path, embeddings)

        print(f"Saved embeddings to {path}")
        print(f"Shape: {embeddings.shape}")

    def load_embeddings(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embeddings file not found: {path}")

        embeddings = np.load(path)

        print(f"Loaded embeddings from {path}")
        print(f"Shape: {embeddings.shape}")

        return embeddings