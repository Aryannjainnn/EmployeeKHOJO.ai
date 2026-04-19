"""
build_index.py — Run once to build all indexes from profiles.csv
Usage: python build_index.py
"""

import sys
import time
import pandas as pd
from pathlib import Path
from indexer import HybridIndex

# Fix Windows console encoding for Unicode
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    csv_path = "data/profiles.csv"
    index_path = Path("data")

    print("\n" + "="*60)
    print("  Component A - Building Hybrid Index")
    print("="*60 + "\n")

    # Load dataset
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[OK] Loaded {len(df)} profiles from {csv_path}")

    # Build index
    idx = HybridIndex()
    idx.build_from_dataframe(df)

    # Save
    idx.save(index_path)
    elapsed = time.time() - t0
    print(f"\n[OK] Done in {elapsed:.1f}s - index saved to {index_path}/")
    print(f"  Files: {[f.name for f in index_path.iterdir()]}")

    return idx

if __name__ == "__main__":
    main()