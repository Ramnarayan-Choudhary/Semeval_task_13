from huggingface_hub import list_datasets
# Try a few likely queries; print top matches so you can pick the exact ID.
queries = [
    "SemEval-2026 Task 13 Task A",
    "SemEval-2026 Task13 A",
    "machine-generated code detection Task A",
    "mbzuai SemEval task 13 A",
]
seen = set()
for q in queries:
    print(f"\n=== Search: {q} ===")
    for d in list_datasets(search=q):
        if d.id in seen: 
            continue
        seen.add(d.id)
        print(d.id)
