"""
update_index.py — Interactive CLI for incrementally updating the index.
Supports: Add new profile, Modify existing profile, Delete profile, Search.
Usage: python update_index.py
"""

import sys
import uuid
import pandas as pd
from pathlib import Path
from indexer import HybridIndex

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

INDEX_PATH = Path("data")
CSV_PATH = "data/profiles.csv"

def load_index():
    if not INDEX_PATH.exists():
        print("  [ERROR] Index not found. Run build_index.py first.")
        sys.exit(1)
    return HybridIndex.load(INDEX_PATH)


def add_profile(idx: HybridIndex):
    """Interactively add a new profile."""
    print("\n--- ADD NEW PROFILE ---")
    
    name = input("  Name: ").strip()
    if not name:
        print("  [CANCELLED] Name is required.")
        return
    
    core_skills = input("  Core Skills (comma-separated): ").strip()
    secondary_skills = input("  Secondary Skills (comma-separated): ").strip()
    soft_skills = input("  Soft Skills (comma-separated): ").strip()
    skill_summary = input("  Skill Summary (a brief paragraph): ").strip()
    potential_roles = input("  Potential Roles (comma-separated): ").strip()
    
    yoe_input = input("  Years of Experience: ").strip()
    try:
        yoe = float(yoe_input) if yoe_input else 0.0
    except ValueError:
        yoe = 0.0

    doc_id = f"MANUAL_{uuid.uuid4().hex[:8].upper()}"
    
    new_row = {
        "id": doc_id,
        "name": name,
        "core_skills": core_skills,
        "secondary_skills": secondary_skills,
        "soft_skills": soft_skills,
        "skill_summary": skill_summary,
        "potential_roles": potential_roles,
        "years_of_experience": yoe,
    }

    print(f"\n  Preview:")
    for k, v in new_row.items():
        print(f"    {k}: {v}")

    confirm = input("\n  Confirm add? (y/n): ").strip().lower()
    if confirm != "y":
        print("  [CANCELLED]")
        return

    # Load existing CSV, append, and run incremental update
    df = pd.read_csv(CSV_PATH)
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    result = idx.update_from_dataframe(new_df)
    idx.save(INDEX_PATH)

    # Also append to CSV so future full rebuilds include this profile
    new_df.to_csv(CSV_PATH, index=False)

    print(f"\n  [OK] Profile '{name}' added (ID: {doc_id})")
    print(f"  Summary: {result}")


def modify_profile(idx: HybridIndex):
    """Search for and modify an existing profile."""
    print("\n--- MODIFY EXISTING PROFILE ---")
    
    search_name = input("  Search by name: ").strip()
    if not search_name:
        print("  [CANCELLED]")
        return

    # Search for candidates
    results = idx.lexical_search(search_name, top_k=5)
    if not results:
        print("  No profiles found.")
        return

    print(f"\n  Found {len(results)} matches:")
    for i, r in enumerate(results, 1):
        name = r.get("name", "?")
        doc_id = r.get("doc_id", "?")
        roles = str(r.get("potential_roles", ""))[:50]
        skills = [s for s, _ in (r.get("top_skills") or [])[:3]]
        print(f"    {i}. [{doc_id}] {name} | {roles} | {skills}")

    choice = input(f"\n  Select profile (1-{len(results)}), or 0 to cancel: ").strip()
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(results):
            print("  [CANCELLED]")
            return
    except ValueError:
        print("  [CANCELLED]")
        return

    selected = results[choice_idx]
    doc_id = selected["doc_id"]
    old_name = selected.get("name", "?")

    print(f"\n  Modifying: {old_name} (ID: {doc_id})")
    print("  Leave blank to keep current value.\n")

    df = pd.read_csv(CSV_PATH)
    row_mask = df["id"].astype(str) == str(doc_id)

    if not row_mask.any():
        print(f"  [ERROR] Profile ID '{doc_id}' not found in CSV.")
        return

    row_idx = df[row_mask].index[0]
    current = df.loc[row_idx]

    fields = ["name", "core_skills", "secondary_skills", "soft_skills",
              "skill_summary", "potential_roles", "years_of_experience"]

    for field in fields:
        current_val = str(current.get(field, ""))
        display_val = current_val[:60] + "..." if len(current_val) > 60 else current_val
        new_val = input(f"  {field} [{display_val}]: ").strip()
        if new_val:
            if field == "years_of_experience":
                try:
                    df.at[row_idx, field] = float(new_val)
                except ValueError:
                    print(f"    Invalid number, keeping: {current_val}")
            else:
                df.at[row_idx, field] = new_val

    confirm = input("\n  Confirm changes? (y/n): ").strip().lower()
    if confirm != "y":
        print("  [CANCELLED]")
        return

    result = idx.update_from_dataframe(df)
    idx.save(INDEX_PATH)
    df.to_csv(CSV_PATH, index=False)

    print(f"\n  [OK] Profile '{old_name}' updated.")
    print(f"  Summary: {result}")


def delete_profile(idx: HybridIndex):
    """Search for and delete a profile."""
    print("\n--- DELETE PROFILE ---")

    search_name = input("  Search by name: ").strip()
    if not search_name:
        print("  [CANCELLED]")
        return

    results = idx.lexical_search(search_name, top_k=5)
    if not results:
        print("  No profiles found.")
        return

    print(f"\n  Found {len(results)} matches:")
    for i, r in enumerate(results, 1):
        name = r.get("name", "?")
        doc_id = r.get("doc_id", "?")
        print(f"    {i}. [{doc_id}] {name}")

    choice = input(f"\n  Select profile to DELETE (1-{len(results)}), or 0 to cancel: ").strip()
    try:
        choice_idx = int(choice) - 1
        if choice_idx < 0 or choice_idx >= len(results):
            print("  [CANCELLED]")
            return
    except ValueError:
        print("  [CANCELLED]")
        return

    selected = results[choice_idx]
    doc_id = selected["doc_id"]
    old_name = selected.get("name", "?")

    confirm = input(f"\n  Are you sure you want to DELETE '{old_name}'? (y/n): ").strip().lower()
    if confirm != "y":
        print("  [CANCELLED]")
        return

    # Remove from CSV
    df = pd.read_csv(CSV_PATH)
    df = df[df["id"].astype(str) != str(doc_id)]
    
    result = idx.update_from_dataframe(df)
    idx.save(INDEX_PATH)
    df.to_csv(CSV_PATH, index=False)

    print(f"\n  [OK] Profile '{old_name}' deleted.")
    print(f"  Summary: {result}")


def quick_search(idx: HybridIndex):
    """Quick test search."""
    print("\n--- QUICK SEARCH ---")
    query = input("  Query: ").strip()
    if not query:
        return

    print("\n  BM25 (Lexical):")
    for r in idx.lexical_search(query, top_k=5):
        name = r.get("name", "?")
        score = r.get("bm25_score", 0)
        skills = [s for s, _ in (r.get("top_skills") or [])[:4]]
        print(f"    [{score:.2f}] {name} | {skills}")

    print("\n  SBERT (Semantic):")
    for r in idx.semantic_search(query, top_k=5):
        name = r.get("name", "?")
        score = r.get("semantic_score", 0)
        skills = [s for s, _ in (r.get("top_skills") or [])[:4]]
        print(f"    [{score:.4f}] {name} | {skills}")


def main():
    print("\n" + "=" * 50)
    print("  Component A — Index Update Manager")
    print("=" * 50)

    idx = load_index()
    print(f"  Index loaded: {len(idx.bm25)} documents\n")

    while True:
        print("-" * 50)
        print("  1. Add new profile")
        print("  2. Modify existing profile")
        print("  3. Delete profile")
        print("  4. Quick search")
        print("  5. Reload index from disk")
        print("  0. Exit")
        print("-" * 50)

        choice = input("  Choose (0-5): ").strip()

        if choice == "1":
            add_profile(idx)
        elif choice == "2":
            modify_profile(idx)
        elif choice == "3":
            delete_profile(idx)
        elif choice == "4":
            quick_search(idx)
        elif choice == "5":
            idx = load_index()
            print(f"  Reloaded: {len(idx.bm25)} documents")
        elif choice == "0":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice.")


if __name__ == "__main__":
    main()
