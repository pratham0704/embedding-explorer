import sys
sys.path.append('.')

from src.model_comparison import ModelComparison

def main():
    print("\n=== F2 MODEL COMPARISON ===\n")
    
    comparison = ModelComparison()
    results = comparison.compare_failure_modes()
    
    print("\n=== COMPARISON COMPLETE ===")

if __name__ == "__main__":
    main()