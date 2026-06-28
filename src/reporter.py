import json
import os
import random


class Reporter:
    def __init__(self, output_dir="data/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_failure_report(
        self,
        tickets,
        embeddings,
        similarity,
        failure_detector,
        sample_size=500,
    ):
        if not isinstance(tickets, list):
            raise TypeError("tickets must be a list")

        if len(tickets) == 0:
            raise ValueError("tickets list cannot be empty")

        if len(tickets) != len(embeddings):
            raise ValueError("tickets and embeddings must have the same length")

        sample_count = min(sample_size, len(tickets))

        sampled_indices = random.sample(
            range(len(tickets)),
            sample_count,
        )

        failures = []
        total_pairs_analyzed = 0

        for i in range(len(sampled_indices)):
            index_a = sampled_indices[i]
            ticket_a = tickets[index_a]
            embedding_a = embeddings[index_a]

            for j in range(i + 1, len(sampled_indices)):
                index_b = sampled_indices[j]
                ticket_b = tickets[index_b]
                embedding_b = embeddings[index_b]

                text_a = ticket_a["text"]
                text_b = ticket_b["text"]

                score = similarity.cosine_similarity(
                    embedding_a,
                    embedding_b,
                )

                detection = failure_detector.detect(
                    text_a=text_a,
                    text_b=text_b,
                    score=score,
                )

                total_pairs_analyzed += 1

                if detection["failure_mode"] is not None:
                    failure = {
                        "failure_mode": detection["failure_mode"],
                        "confidence": detection["confidence"],
                        "score": detection["score"],
                        "text_a": text_a,
                        "text_b": text_b,
                        "category_a": ticket_a.get("category"),
                        "category_b": ticket_b.get("category"),
                        "explanation": detection["explanation"],
                    }

                    failures.append(failure)

        failures_by_mode = {
            "IDENTIFIER_CONFUSION": 0,
            "NEGATION_BLINDNESS": 0,
            "DOMAIN_VOCABULARY_GAP": 0,
        }

        failures_by_category = {}

        examples = {
            "IDENTIFIER_CONFUSION": [],
            "NEGATION_BLINDNESS": [],
            "DOMAIN_VOCABULARY_GAP": [],
        }

        for failure in failures:
            mode = failure["failure_mode"]

            if mode not in failures_by_mode:
                failures_by_mode[mode] = 0

            failures_by_mode[mode] += 1

            category_pair = f"{failure['category_a']} <-> {failure['category_b']}"

            if category_pair not in failures_by_category:
                failures_by_category[category_pair] = 0

            failures_by_category[category_pair] += 1

        for mode in examples:
            mode_failures = [
                failure for failure in failures
                if failure["failure_mode"] == mode
            ]

            deduped_failures = []
            seen_pairs = set()

            for failure in mode_failures:
                text_a = failure["text_a"]
                text_b = failure["text_b"]

                # Order-independent key:
                # A-B and B-A should be treated as the same example.
                pair_key = tuple(sorted([text_a, text_b]))

                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)
                deduped_failures.append(failure)

            deduped_failures.sort(
                key=lambda failure: failure["score"],
                reverse=True,
            )

            examples[mode] = deduped_failures[:3]   

        total_failures_detected = len(failures)

        if total_pairs_analyzed == 0:
            failure_rate = 0.0
        else:
            failure_rate = total_failures_detected / total_pairs_analyzed

        report = {
            "total_pairs_analyzed": total_pairs_analyzed,
            "total_failures_detected": total_failures_detected,
            "failure_rate": failure_rate,
            "failures_by_mode": failures_by_mode,
            "failures_by_category": failures_by_category,
            "examples": examples,
        }

        self.print_report(report)

        return report

    def print_report(self, report):
        total_pairs = report["total_pairs_analyzed"]
        total_failures = report["total_failures_detected"]
        failure_rate_percent = report["failure_rate"] * 100

        print("\n=== EMBEDDING FAILURE CATALOGUE ===\n")

        print(f"Total pairs analyzed:    {total_pairs}")
        print(f"Total failures detected: {total_failures}")
        print(f"Failure rate:            {failure_rate_percent:.2f}%")

        print("\nFailures by mode:")

        for mode, count in report["failures_by_mode"].items():
            if total_failures == 0:
                mode_percent = 0.0
            else:
                mode_percent = (count / total_failures) * 100

            print(f"  {mode}: {count} ({mode_percent:.2f}%)")

        if "failures_by_category" in report:
            print("\nFailures by category pair:")

            sorted_categories = sorted(
                report["failures_by_category"].items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for category_pair, count in sorted_categories[:10]:
                print(f"  {category_pair}: {count}")

        print("\nExamples:")

        for mode, examples in report["examples"].items():
            print(f"\n{mode}:")

            if not examples:
                print("  No examples found.")
                continue

            for example in examples:
                print("-" * 100)
                print(f"  Score:       {example['score']:.4f}")
                print(f"  Confidence:  {example['confidence']}")
                print(f"  Category A:  {example['category_a']}")
                print(f"  Category B:  {example['category_b']}")
                print(f"  Text A:      {example['text_a']}")
                print(f"  Text B:      {example['text_b']}")
                print(f"  Explanation: {example['explanation']}")

    def save_report(self, report, filename="failure_report.json"):
        path = os.path.join(self.output_dir, filename)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        print(f"\nSaved failure report to {path}")

        return path