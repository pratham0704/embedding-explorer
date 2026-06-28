from sentence_transformers import SentenceTransformer
import numpy as np


class ModelComparison:
    def __init__(self, models=None):
        self.model_names = models or [
            "all-MiniLM-L6-v2",       # small, fast, 384 dims
            "all-mpnet-base-v2",      # larger, better quality, 768 dims
            "paraphrase-MiniLM-L6-v2" # trained for paraphrase detection
        ]

        self.models = {}

        self.similarity_threshold = 0.70

    def _load_model(self, model_name):
        if model_name not in self.models:
            print(f"Loading {model_name}...")
            self.models[model_name] = SentenceTransformer(model_name)

        return self.models[model_name]

    def compare_pair(self, text_a, text_b):
        scores = {}

        for model_name in self.model_names:
            model = self._load_model(model_name)

            embeddings = model.encode(
                [text_a, text_b],
                convert_to_numpy=True,
            )

            score = self._cosine_similarity(
                embeddings[0],
                embeddings[1],
            )

            scores[model_name] = float(score)

        high_low_labels = {}

        for model_name, score in scores.items():
            if score >= self.similarity_threshold:
                high_low_labels[model_name] = "high"
            else:
                high_low_labels[model_name] = "low"

        unique_labels = set(high_low_labels.values())

        agreement = len(unique_labels) == 1

        disagreement_signal = None

        if not agreement:
            high_models = []
            low_models = []

            for model_name, label in high_low_labels.items():
                if label == "high":
                    high_models.append(model_name)
                else:
                    low_models.append(model_name)

            disagreement_signal = (
                f"{', '.join(high_models)} think high; "
                f"{', '.join(low_models)} think low"
            )

        return {
            "text_a": text_a,
            "text_b": text_b,
            "scores": scores,
            "agreement": agreement,
            "disagreement_signal": disagreement_signal,
        }

    def compare_failure_modes(self):
        test_cases = {
            "identifier_confusion": [
                ("CASE-2024-000123", "CASE-2024-000124"),
                ("TXN-987654321", "TXN-987654322"),
            ],
            "negation_blindness": [
                (
                    "The transaction is suspicious.",
                    "The transaction is not suspicious.",
                ),
                (
                    "The KYC verification is complete.",
                    "The KYC verification is not complete.",
                ),
                (
                    "The customer is eligible for the loan.",
                    "The customer is not eligible for the loan.",
                ),
            ],
            "domain_vocabulary": [
                (
                    "NACH mandate rejected",
                    "auto debit instruction cancelled",
                ),
                (
                    "NPA account detected",
                    "non performing asset found",
                ),
                (
                    "CIBIL score low",
                    "credit score below threshold",
                ),
            ],
            "semantic_vs_lexical": [
                (
                    "The payment was approved.",
                    "The payment was not approved.",
                ),
                (
                    "The customer is eligible.",
                    "The borrower qualifies for credit.",
                ),
            ],
        }

        results = {}

        print("\n=== MODEL COMPARISON REPORT ===")
        print(f"Similarity threshold: {self.similarity_threshold}")

        for failure_mode, pairs in test_cases.items():
            print("\n" + "=" * 100)
            print(f"Failure mode test: {failure_mode}")
            print("=" * 100)

            results[failure_mode] = []

            for text_a, text_b in pairs:
                comparison = self.compare_pair(text_a, text_b)
                results[failure_mode].append(comparison)

                self._print_comparison(comparison)

        return results

    def _cosine_similarity(self, vec_a, vec_b):
        vec_a = np.asarray(vec_a)
        vec_b = np.asarray(vec_b)

        dot_product = np.sum(vec_a * vec_b)

        norm_a = np.sqrt(np.sum(vec_a * vec_a))
        norm_b = np.sqrt(np.sum(vec_b * vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _print_comparison(self, comparison):
        print("\n" + "-" * 100)
        print(f"Text A: {comparison['text_a']}")
        print(f"Text B: {comparison['text_b']}")
        print("\nScores:")

        for model_name, score in comparison["scores"].items():
            label = "HIGH" if score >= self.similarity_threshold else "LOW"
            print(f"  {model_name:<30} {score:.4f}  [{label}]")

        print(f"\nAgreement: {comparison['agreement']}")

        if comparison["disagreement_signal"]:
            print(f"Disagreement: {comparison['disagreement_signal']}")