import sys
sys.path.append('.')

from src.dataset import Dataset

SAVE_PATH = "data/tickets/fintech_tickets_10k.json"

def main():
    print("Generating 10,000 fintech support tickets...")
    
    dataset = Dataset()
    tickets = dataset.generate(n=10000, save_path=SAVE_PATH)
    
    # Print stats
    total = len(tickets)
    
    with_identifier = sum(1 for t in tickets if t["metadata"]["has_identifier"])
    with_negation = sum(1 for t in tickets if t["metadata"]["has_negation"])
    with_domain = sum(1 for t in tickets if t["metadata"]["domain_terms"])
    
    category_counts = {}
    for t in tickets:
        cat = t["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nDataset Stats:")
    print(f"  Total tickets:         {total}")
    print(f"  With identifiers:      {with_identifier} ({with_identifier/total*100:.1f}%)")
    print(f"  With negation:         {with_negation} ({with_negation/total*100:.1f}%)")
    print(f"  With domain terms:     {with_domain} ({with_domain/total*100:.1f}%)")
    print(f"\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat:25} {count}")
    print(f"\nSaved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()