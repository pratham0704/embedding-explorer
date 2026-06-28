import re


class FailureDetector:
    def __init__(self):
        self.identifier_pattern = re.compile(
            r"\b(ERR_[A-Z_]+_\d+|TXN-\d+|CASE-\d{4}-\d+)\b"
        )
        self.domain_equivalents = {
            "NACH": ["auto debit", "standing instruction", "automatic payment"],
            "KYC": ["identity verification", "document verification", "know your customer"],
            "NPA": ["non performing asset", "bad loan", "defaulted account"],
            "CIBIL": ["credit score", "credit bureau", "credit report"],
            "NBFC": ["non banking financial", "financial company", "lending company"],
        }

        self.negation_words = {
            "not",
            "no",
            "never",
            "rejected",
            "denied",
            "failed",
            "declined",
            "unable",
            "cannot",
            "blocked",
            "inactive",
            "failure",
            "invalid",
            "incorrect",
            "incomplete",
            "mismatch",
        }

        self.domain_terms = {
            "NACH",
            "NPA",
            "KYC",
            "CIBIL",
            "NBFC",
            "UPI",
            "NEFT",
            "IMPS",
            "AML",
            "NPCI",
        }

        self.high_similarity_threshold = 0.85
        self.low_similarity_threshold = 0.20

    def detect(self, text_a, text_b, score):
        identifiers_a = self._extract_identifiers(text_a)
        identifiers_b = self._extract_identifiers(text_b)

        has_negation_a = self._has_negation(text_a)
        has_negation_b = self._has_negation(text_b)

        domain_terms_a = self._extract_domain_terms(text_a)
        domain_terms_b = self._extract_domain_terms(text_b)

        lexical_overlap = self._lexical_overlap(text_a, text_b)

        # 1. IDENTIFIER_CONFUSION
        if (
            score >= self.high_similarity_threshold
            and identifiers_a
            and identifiers_b
            and identifiers_a != identifiers_b
        ):
            return {
                "failure_mode": "IDENTIFIER_CONFUSION",
                "confidence": "high",
                "explanation": (
                    "High similarity between different identifiers. "
                    "Dense retrieval cannot reliably distinguish IDs like case numbers, "
                    "transaction IDs, or error codes."
                ),
                "score": float(score),
            }

        # 2. NEGATION_BLINDNESS
        if score >= self.high_similarity_threshold:
            one_has_negation = has_negation_a != has_negation_b
            both_have_negation_but_conflicting = self._has_conflicting_negation_state(
                text_a,
                text_b,
            )

            if one_has_negation or both_have_negation_but_conflicting:
                return {
                    "failure_mode": "NEGATION_BLINDNESS",
                    "confidence": "high",
                    "explanation": (
                        "High similarity despite negation difference. "
                        "The model is treating negated and non-negated statements as similar."
                    ),
                    "score": float(score),
                }

        # 3. DOMAIN_VOCABULARY_GAP
        if score <= 0.50 and self._has_domain_equivalence_gap(text_a, text_b):
            return {
                "failure_mode": "DOMAIN_VOCABULARY_GAP",
                "confidence": "high",
                "explanation": (
                    "Low similarity between a domain-specific term and its plain-English "
                    "equivalent. The model may not understand this banking terminology."
                ),
                "score": float(score),
            }

        # 4. SEMANTIC_LEXICAL
        #
        # This is intentionally conservative.
        # High lexical overlap + high score can be suspicious, but we cannot reliably prove
        # different meaning without deeper NLP/LLM/rule-based intent checks.
        #
        # So for now we do NOT label this as a failure mode.
        if (
            score >= self.high_similarity_threshold
            and lexical_overlap >= 0.70
        ):
            return {
                "failure_mode": None,
                "confidence": "low",
                "explanation": (
                    "No automatic failure mode confirmed. However, the texts have high "
                    "lexical overlap and high similarity, so semantic-vs-lexical confusion "
                    "may need manual review."
                ),
                "score": float(score),
            }

        # 5. No failure mode detected
        return {
            "failure_mode": None,
            "confidence": "low",
            "explanation": "No obvious embedding failure mode detected.",
            "score": float(score),
        }

    def _extract_identifiers(self, text):
        matches = self.identifier_pattern.findall(text)
        return set(matches)

    def _has_negation(self, text):
        words = self._tokenize_words(text)

        for word in words:
            if word in self.negation_words:
                return True

        return False

    def _extract_domain_terms(self, text):
        found_terms = set()
        upper_text = text.upper()

        for term in self.domain_terms:
            pattern = rf"\b{re.escape(term)}\b"

            if re.search(pattern, upper_text):
                found_terms.add(term)

        return found_terms

    def _lexical_overlap(self, text_a, text_b):
        words_a = set(self._tokenize_words(text_a))
        words_b = set(self._tokenize_words(text_b))

        if not words_a and not words_b:
            return 0.0

        total_unique_words = words_a.union(words_b)
        shared_words = words_a.intersection(words_b)

        if not total_unique_words:
            return 0.0

        return len(shared_words) / len(total_unique_words)

    def _tokenize_words(self, text):
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def _has_conflicting_negation_state(self, text_a, text_b):
        """
        Simple heuristic for cases like:

        - "approved" vs "rejected"
        - "active" vs "inactive"
        - "complete" vs "incomplete"
        - "valid" vs "invalid"

        This is not perfect semantic reasoning.
        It is deliberately narrow and explainable.
        """

        words_a = set(self._tokenize_words(text_a))
        words_b = set(self._tokenize_words(text_b))

        opposite_pairs = [
            ("approved", "rejected"),
            ("approved", "denied"),
            ("approved", "declined"),
            ("eligible", "ineligible"),
            ("eligible", "not"),
            ("active", "inactive"),
            ("complete", "incomplete"),
            ("valid", "invalid"),
            ("correct", "incorrect"),
            ("success", "failed"),
            ("successful", "failed"),
            ("verified", "failed"),
            ("authorized", "unauthorized"),
            ("compliant", "noncompliant"),
        ]

        for positive_word, negative_word in opposite_pairs:
            a_has_positive = positive_word in words_a
            a_has_negative = negative_word in words_a

            b_has_positive = positive_word in words_b
            b_has_negative = negative_word in words_b

            if a_has_positive and b_has_negative:
                return True

            if a_has_negative and b_has_positive:
                return True

        return False

    def _has_domain_equivalence_gap(self, text_a, text_b):
        upper_a = text_a.upper()
        upper_b = text_b.upper()

        lower_a = text_a.lower()
        lower_b = text_b.lower()

        for term, equivalents in self.domain_equivalents.items():
            term_in_a = re.search(rf"\b{re.escape(term)}\b", upper_a) is not None
            term_in_b = re.search(rf"\b{re.escape(term)}\b", upper_b) is not None

            equivalent_in_a = any(
                equivalent in lower_a
                for equivalent in equivalents
            )

            equivalent_in_b = any(
                equivalent in lower_b
                for equivalent in equivalents
            )

            if term_in_a and equivalent_in_b:
                return True

            if term_in_b and equivalent_in_a:
                return True

        return False