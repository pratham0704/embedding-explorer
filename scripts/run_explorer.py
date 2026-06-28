import os
import sys

sys.path.append(".")

from src.dataset import Dataset
from src.embedder import Embedder
from src.similarity import Similarity
from src.failure_detector import FailureDetector
from src.reporter import Reporter


TICKETS_PATH = "data/tickets/fintech_tickets_10k.json"
EMBEDDINGS_PATH = "data/embeddings/tickets_minilm.npy"
REPORT_FILENAME = "failure_report.json"


def main():
    print("\n=== F2 EMBEDDING FAILURE EXPLORER ===\n")

    # Step 1: Load tickets
    print(f"Loading tickets from {TICKETS_PATH}...")

    dataset = Dataset()
    tickets = dataset.load(TICKETS_PATH)

    print(f"Loaded {len(tickets)} tickets.")

    # Step 2: Load or create embeddings
    embedder = Embedder(model_name="all-MiniLM-L6-v2")

    if os.path.exists(EMBEDDINGS_PATH):
        print(f"\nEmbeddings found at {EMBEDDINGS_PATH}.")
        embeddings = embedder.load_embeddings(EMBEDDINGS_PATH)
    else:
        print(f"\nNo embeddings found at {EMBEDDINGS_PATH}.")
        print("Creating embeddings now...")

        embeddings = embedder.embed_tickets(
            tickets=tickets,
            batch_size=64,
        )

        embedder.save_embeddings(
            embeddings=embeddings,
            path=EMBEDDINGS_PATH,
        )

    # Defensive sanity check
    if len(tickets) != len(embeddings):
        raise ValueError(
            f"Ticket count and embedding count mismatch: "
            f"{len(tickets)} tickets vs {len(embeddings)} embeddings"
        )

    # Step 3: Initialize components
    print("\nInitializing similarity and failure detector...")

    similarity = Similarity()
    failure_detector = FailureDetector()
    reporter = Reporter(output_dir="data/reports")

    # Step 4: Run failure report
    print("\nRunning failure analysis with sample_size=500...")

    report = reporter.generate_failure_report(
        tickets=tickets,
        embeddings=embeddings,
        similarity=similarity,
        failure_detector=failure_detector,
        sample_size=500,
    )

    # Step 5: Save report
    reporter.save_report(
        report=report,
        filename=REPORT_FILENAME,
    )

    # Step 6: Completion message
    print("\n=== F2 PIPELINE COMPLETE ===")
    print(f"Report saved to: data/reports/{REPORT_FILENAME}")


if __name__ == "__main__":
    main()