"""
Deep analysis of NEqCor tables to understand symmetry operations encoding.
"""
import numpy as np
import re

def extract_neqcor_table(log_file: str, occurrence: int = 0) -> tuple:
    """Extract NEqCor table from Gaussian log."""
    with open(log_file, 'r') as f:
        content = f.read()

    # Find all NEqCor sections
    pattern = r'NEqCor:\s*\n((?:\s+\d+(?:\s+[-]?\d+)+\s*\n)+)'
    matches = list(re.finditer(pattern, content))

    if occurrence >= len(matches):
        return None, None

    match = matches[occurrence]
    table_text = match.group(1)

    # Parse table
    rows = []
    for line in table_text.strip().split('\n'):
        parts = [int(x) for x in line.split()]
        rows.append(parts)

    # Determine structure
    num_cols = len(rows[0]) if rows else 0
    num_rows = len(rows)

    return np.array(rows), num_cols

def analyze_nh3_neqcor(table: np.ndarray):
    """
    Analyze NH3 NEqCor table structure.
    NH3: 4 atoms, 12 coordinates
    Atom 1 (N):  coords 1,2,3   (X_N, Y_N, Z_N)
    Atom 2 (H1): coords 4,5,6   (X_H1, Y_H1, Z_H1)
    Atom 3 (H2): coords 7,8,9   (X_H2, Y_H2, Z_H2)
    Atom 4 (H3): coords 10,11,12 (X_H3, Y_H3, Z_H3)
    """
    print("\n" + "=" * 70)
    print("NH3 NEqCor ANALYSIS")
    print("=" * 70)

    print("\nTable structure:")
    print(f"  Rows: {table.shape[0]}")
    print(f"  Columns: {table.shape[1]}")

    if table.shape[1] == 2:
        print("\n  Format: [coord_index, transformed_coord]")
        print("  → Represents ONE symmetry operation")

    print("\nDetailed mapping:")
    coord_labels = [
        "X_N", "Y_N", "Z_N",
        "X_H1", "Y_H1", "Z_H1",
        "X_H2", "Y_H2", "Z_H2",
        "X_H3", "Y_H3", "Z_H3"
    ]

    for i in range(12):
        idx = table[i, 0]
        target = table[i, 1]

        sign = "+" if target > 0 else "-"
        target_abs = abs(target)

        source_label = coord_labels[i]
        target_label = coord_labels[target_abs - 1]

        print(f"  {idx:2d} → {target:3d}:  {source_label:6s} → {sign}{target_label:6s}")

    # Infer operation type
    print("\n" + "-" * 70)
    print("OPERATION INFERENCE:")

    # Check X components
    x_pattern = [table[0,1], table[3,1], table[6,1], table[9,1]]  # X_N, X_H1, X_H2, X_H3
    y_pattern = [table[1,1], table[4,1], table[7,1], table[10,1]] # Y components
    z_pattern = [table[2,1], table[5,1], table[8,1], table[11,1]] # Z components

    print(f"  X coords: {x_pattern}")
    print(f"  Y coords: {y_pattern}")
    print(f"  Z coords: {z_pattern}")

    # Check for reflection in YZ plane (X → -X, Y → Y, Z → Z)
    if (x_pattern == [-1, -4, -10, -7] and
        y_pattern == [2, 5, 11, 8] and
        z_pattern == [3, 6, 12, 9]):
        print("\n  ✓ This is σv (reflection in YZ plane)")
        print("    - X components change sign")
        print("    - Y and Z components unchanged")
        print("    - H2 (coords 7-9) ↔ H3 (coords 10-12)")

        # Derive nuclear permutation
        perm = [0, 0, 0, 0]
        perm[0] = 0  # N stays
        perm[1] = 1  # H1 stays
        perm[2] = 3  # H2 → H3
        perm[3] = 2  # H3 → H2

        print(f"\n  Nuclear permutation: {perm}")
        print("    Atom 0 (N)  → Atom 0")
        print("    Atom 1 (H1) → Atom 1")
        print("    Atom 2 (H2) → Atom 3")
        print("    Atom 3 (H3) → Atom 2")

        # Derive transformation matrix
        T = np.array([
            [-1, 0, 0],  # X → -X
            [0, 1, 0],   # Y → Y
            [0, 0, 1]    # Z → Z
        ])
        print(f"\n  Transformation matrix T:")
        for row in T:
            print(f"    {row}")

        return perm, T

    return None, None


def analyze_h2o_neqcor():
    """
    Analyze H2O NEqCor table (from user's example).
    H2O: 3 atoms, 9 coordinates
    C2v: 4 operations (E, C2, σv, σ'v)
    """
    print("\n" + "=" * 70)
    print("H2O NEqCor ANALYSIS (from user example)")
    print("=" * 70)

    # User provided this table
    table = np.array([
        [1, -1, -1,  1],   # X_O
        [2, -2,  2, -2],   # Y_O
        [3,  3,  3,  3],   # Z_O
        [4, -7, -4,  7],   # X_H1
        [5, -8,  5, -8],   # Y_H1
        [6,  9,  6,  9],   # Z_H1
        [7, -4, -7,  4],   # X_H2
        [8, -5,  8, -5],   # Y_H2
        [9,  6,  9,  6]    # Z_H2
    ])

    print("\nTable structure:")
    print(f"  Rows: 9 (3 atoms × 3 coords)")
    print(f"  Columns: 4 (E, C2, σv, σ'v)")

    operations = ["E", "C2", "σv(yz)", "σ'v(xz)"]
    coord_labels = ["X_O", "Y_O", "Z_O", "X_H1", "Y_H1", "Z_H1", "X_H2", "Y_H2", "Z_H2"]

    print("\n" + "-" * 70)
    print("TRANSFORMATION UNDER EACH OPERATION:")

    for op_idx, op_name in enumerate(operations):
        print(f"\n{op_name}:")
        for coord_idx in range(9):
            source = coord_idx + 1
            target = table[coord_idx, op_idx]

            sign = "+" if target > 0 else "-"
            target_abs = abs(target)

            source_label = coord_labels[coord_idx]
            target_label = coord_labels[target_abs - 1]

            print(f"  {source:2d} → {target:3d}:  {source_label:6s} → {sign}{target_label:6s}")

    print("\n" + "-" * 70)
    print("DERIVING SYMMETRY OPERATIONS:")

    for op_idx, op_name in enumerate(operations):
        print(f"\n{op_name}:")

        # Extract transformation for this operation
        transform_map = {}
        for coord_idx in range(9):
            source = coord_idx + 1
            target = table[coord_idx, op_idx]
            transform_map[source] = target

        # Derive nuclear permutation
        perm = [0, 0, 0]
        for atom in range(3):
            # Check where X coordinate of this atom goes
            x_coord = atom * 3 + 1
            x_target = abs(transform_map[x_coord])
            target_atom = (x_target - 1) // 3
            perm[atom] = target_atom

        print(f"  Nuclear permutation: {perm}")

        # Derive transformation matrix
        T = np.zeros((3, 3))
        # Look at how O transforms (first 3 coords)
        for i in range(3):
            source = i + 1
            target = transform_map[source]
            sign = 1 if target > 0 else -1
            target_coord = (abs(target) - 1) % 3
            T[target_coord, i] = sign

        print(f"  Transformation matrix:")
        for row in T:
            print(f"    {row}")


# Main execution
print("=" * 70)
print("NEqCor TABLE ANALYSIS FOR SYMMETRY OPERATIONS")
print("=" * 70)

# Analyze NH3
log_file = "ErrorDirectory/FAKE_FREQ/fake_freq_7.log"
nh3_table, nh3_cols = extract_neqcor_table(log_file, occurrence=0)

if nh3_table is not None:
    print(f"\n✓ Extracted NH3 NEqCor table ({nh3_table.shape[0]} rows, {nh3_cols} columns)")
    perm, T = analyze_nh3_neqcor(nh3_table)
else:
    print("\n✗ Could not extract NH3 table")

# Analyze H2O
analyze_h2o_neqcor()

print("\n" + "=" * 70)
print("CONCLUSIONS:")
print("=" * 70)
print("""
1. NEqCor encodes coordinate transformations under symmetry operations
2. Format depends on number of operations shown:
   - 2 columns (NH3): Shows ONE operation (e.g., σv)
   - 4 columns (H2O): Shows ALL 4 operations of C2v

3. We can derive:
   ✓ Nuclear permutations (which atoms swap)
   ✓ Transformation matrices (how coordinates transform)

4. This is MORE DIRECT than force-based inference!
   - No need to try 60+ rotation matrices
   - No need to compute RMSD
   - Just parse the table and extract operations

5. Limitation: Only shows operations with non-trivial nuclear permutations?
   - NH3 C3v has 6 operations, but table shows only 1
   - May need to combine with force-based inference for complete set
""")
print("=" * 70)
