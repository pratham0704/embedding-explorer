import numpy as np


class Similarity:
    def __init__(self):
        # Thresholds based on what we measured in exercises
        self.thresholds = {
            "identical": 0.95,    # likely same text or near-duplicate
            "very_high": 0.85,    # high similarity — could be failure mode
            "high": 0.70,         # meaningful similarity
            "moderate": 0.50,     # weak similarity
            "low": 0.30,          # near unrelated
        }

    def cosine_similarity(self, vec_a, vec_b):
        vec_a = np.asarray(vec_a)
        vec_b = np.asarray(vec_b)

        dot_product = np.sum(vec_a * vec_b)

        norm_a = np.sqrt(np.sum(vec_a * vec_a))
        norm_b = np.sqrt(np.sum(vec_b * vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def score_pair(self, text_a, text_b, vec_a, vec_b):
        score = self.cosine_similarity(vec_a, vec_b)

        interpretation = self._interpret_score(score)
        warning = self._build_warning(score)

        return {
            "text_a": text_a,
            "text_b": text_b,
            "score": float(score),
            "interpretation": interpretation,
            "warning": warning,
        }

    def score_matrix(self, texts, embeddings):
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        embeddings = np.asarray(embeddings)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D numpy array")

        # Normalize all vectors at once
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Prevent division by zero
        norms[norms == 0] = 1.0

        normalized = embeddings / norms

        # Dot product of normalized vectors = cosine similarity
        matrix = np.dot(normalized, normalized.T)

        return matrix

    def top_k_similar(
        self,
        query_text,
        query_embedding,
        corpus_texts,
        corpus_embeddings,
        k=5,
    ):
        if len(corpus_texts) != len(corpus_embeddings):
            raise ValueError("corpus_texts and corpus_embeddings must have the same length")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        query_embedding = np.asarray(query_embedding)
        corpus_embeddings = np.asarray(corpus_embeddings)

        similarities = []

        for index, corpus_embedding in enumerate(corpus_embeddings):
            score = self.cosine_similarity(query_embedding, corpus_embedding)

            similarities.append(
                {
                    "text": corpus_texts[index],
                    "score": float(score),
                }
            )

        similarities.sort(key=lambda item: item["score"], reverse=True)

        top_results = similarities[:k]

        for rank, result in enumerate(top_results, start=1):
            result["rank"] = rank

        return top_results

    def _interpret_score(self, score):
        if score >= self.thresholds["identical"]:
            return "identical"

        if score >= self.thresholds["very_high"]:
            return "very_high"

        if score >= self.thresholds["high"]:
            return "high"

        if score >= self.thresholds["moderate"]:
            return "moderate"

        return "low"

    def _build_warning(self, score):
        if score >= self.thresholds["identical"]:
            return "Near-duplicate detected. Verify these are not the same document."

        if score >= self.thresholds["very_high"]:
            return "Very high similarity. Check for negation or identifier confusion."

        return None