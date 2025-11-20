"""
Check all NEqCor occurrences in the log to see if they represent different operations.
"""
import numpy as np
import re

log_file = "ErrorDirectory/FAKE_FREQ/fake_freq_7.log"

with open(log_file, 'r') as f:
    content = f.read()

# Find all NEqCor sections
pattern = r'NEqCor:\s*\n((?:\s+\d+(?:\s+[-]?\d+)+\s*\n)+)'
matches = list(re.finditer(pattern, content))

print("=" * 70)
print(f"Found {len(matches)} NEqCor tables in {log_file}")
print("=" * 70)

tables = []
for i, match in enumerate(matches):
    table_text = match.group(1)

    # Parse table
    rows = []
    for line in table_text.strip().split('\n'):
        parts = [int(x) for x in line.split()]
        rows.append(parts)

    table = np.array(rows)
    tables.append(table)

    print(f"\nTable {i+1} (position {match.start()}):")
    print(f"  Shape: {table.shape}")
    print(f"  First 3 rows:")
    for j in range(min(3, len(table))):
        print(f"    {table[j]}")

# Check if all tables are identical
print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)

all_same = True
for i in range(1, len(tables)):
    if not np.array_equal(tables[0], tables[i]):
        all_same = False
        print(f"\nX Table {i+1} differs from Table 1!")

        # Show differences
        diff_rows = []
        for row_idx in range(len(tables[0])):
            if not np.array_equal(tables[0][row_idx], tables[i][row_idx]):
                diff_rows.append(row_idx)

        print(f"  Different rows: {diff_rows[:5]}{'...' if len(diff_rows) > 5 else ''}")

if all_same:
    print("\nAll NEqCor tables are IDENTICAL")
    print("  → Only ONE symmetry operation is encoded")
    print("  → This is likely a 'generator' operation")
    print("  → Other operations must be inferred by composition")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("""
For NH3 C3v (6 operations):
- NEqCor shows: 1 operation (σv reflection)
- Missing: E (identity), C3, C3², σ'v, σ''v

Possible strategies:
1. Use NEqCor for the one operation shown
2. Infer remaining operations via:
   a) Force-based RMSD matching (current method) ✓
   b) Group theory composition (σv * σv = E, etc.)
   c) Combination of both

Recommendation:
- Parse NEqCor to get at least ONE operation efficiently
- Use force-based inference for remaining operations
- This is a hybrid approach: direct + inferred
""")
print("=" * 70)
