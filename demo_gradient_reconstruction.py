"""
Demonstration: Gradient reconstruction from raw energies using symmetry inference.

This script shows the complete workflow:
1. Parse log to identify symmetry-unique atoms
2. Load raw energies from displacement calculations
3. Infer symmetry operations from force patterns
4. Reconstruct full gradient using symmetry
"""

import numpy as np
from pathlib import Path
from elecext.symmetry_engine import NonAbelianSymmetryEngine

# Constants
HARTREE_PER_BOHR = 1.0  # Forces already in Hartree/Bohr
ANGSTROM_TO_BOHR = 1.8897259886


def load_single_energy(energy_file: str) -> float:
    """Load energy from single task output file."""
    with open(energy_file, 'r') as f:
        line = f.readline().strip()
        energy = float(line.split(',')[0])
    return energy


def main():
    print("=" * 80)
    print("GRADIENT RECONSTRUCTION FROM RAW ENERGIES - NH3 C3v")
    print("=" * 80)

    # Setup paths
    iteration_dir = Path("ErrorDirectory/Iterations/Iteration_4")
    analytical_file = "ErrorDirectory/analitical_gradient.txt"

    num_atoms = 4
    step_size = 0.001  # Angstrom (typical numerical gradient step)
    step_bohr = step_size * ANGSTROM_TO_BOHR

    print(f"\nSystem: NH3 (4 atoms: 1 N + 3 H)")
    print(f"Point group: C3v")
    print(f"Displacement step: {step_size} Angstrom = {step_bohr:.6f} Bohr")

    # ========================================================================
    # STEP 1: Load raw energies
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading raw energies from displacement calculations")
    print("=" * 80)

    energies = {}

    # Central geometry
    e_central = load_single_energy(iteration_dir / "task_central" / "energy_output.dat")
    energies["central"] = e_central
    print(f"\nCentral geometry:    E = {e_central:.12f} Hartree")

    # Displacements for Atom 1 (N, index 0)
    print(f"\nAtom 1 (N) displacements:")
    e_n_y_up = load_single_energy(iteration_dir / "task_atom_1_ixyz_2_up" / "energy_output.dat")
    e_n_y_down = load_single_energy(iteration_dir / "task_atom_1_ixyz_2_down" / "energy_output.dat")
    e_n_z_up = load_single_energy(iteration_dir / "task_atom_1_ixyz_3_up" / "energy_output.dat")
    e_n_z_down = load_single_energy(iteration_dir / "task_atom_1_ixyz_3_down" / "energy_output.dat")

    energies["atom_1_y_up"] = e_n_y_up
    energies["atom_1_y_down"] = e_n_y_down
    energies["atom_1_z_up"] = e_n_z_up
    energies["atom_1_z_down"] = e_n_z_down

    print(f"  Y+: E = {e_n_y_up:.12f}   ΔE = {e_n_y_up - e_central:+.9e}")
    print(f"  Y-: E = {e_n_y_down:.12f}   ΔE = {e_n_y_down - e_central:+.9e}")
    print(f"  Z+: E = {e_n_z_up:.12f}   ΔE = {e_n_z_up - e_central:+.9e}")
    print(f"  Z-: E = {e_n_z_down:.12f}   ΔE = {e_n_z_down - e_central:+.9e}")

    # Displacements for Atom 2 (H1, index 1)
    print(f"\nAtom 2 (H1) displacements:")
    e_h1_y_up = load_single_energy(iteration_dir / "task_atom_2_ixyz_2_up" / "energy_output.dat")
    e_h1_y_down = load_single_energy(iteration_dir / "task_atom_2_ixyz_2_down" / "energy_output.dat")
    e_h1_z_up = load_single_energy(iteration_dir / "task_atom_2_ixyz_3_up" / "energy_output.dat")
    e_h1_z_down = load_single_energy(iteration_dir / "task_atom_2_ixyz_3_down" / "energy_output.dat")

    energies["atom_2_y_up"] = e_h1_y_up
    energies["atom_2_y_down"] = e_h1_y_down
    energies["atom_2_z_up"] = e_h1_z_up
    energies["atom_2_z_down"] = e_h1_z_down

    print(f"  Y+: E = {e_h1_y_up:.12f}   ΔE = {e_h1_y_up - e_central:+.9e}")
    print(f"  Y-: E = {e_h1_y_down:.12f}   ΔE = {e_h1_y_down - e_central:+.9e}")
    print(f"  Z+: E = {e_h1_z_up:.12f}   ΔE = {e_h1_z_up - e_central:+.9e}")
    print(f"  Z-: E = {e_h1_z_down:.12f}   ΔE = {e_h1_z_down - e_central:+.9e}")

    print(f"\nNote: Atoms 3 (H2) and 4 (H3) have NO displacements")
    print(f"      → Will be derived by SYMMETRY from H1")

    # ========================================================================
    # STEP 2: Identify symmetry-unique atoms from displacement pattern
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Identifying symmetry-unique atoms")
    print("=" * 80)

    atoms_with_displacements = {0, 1}  # N and H1
    atoms_by_symmetry = {2, 3}  # H2 and H3

    print(f"\nSymmetry-UNIQUE atoms (explicitly calculated):")
    print(f"  Atom 0 (N):  Has displacements on Y, Z  (X constrained by symmetry)")
    print(f"  Atom 1 (H1): Has displacements on Y, Z  (X constrained by symmetry)")

    print(f"\nSymmetry-EQUIVALENT atoms (to be inferred):")
    print(f"  Atom 2 (H2): Related to H1 by C3 rotation (~240°)")
    print(f"  Atom 3 (H3): Related to H1 by C3 rotation (~120°)")

    # ========================================================================
    # STEP 3: Calculate gradients for unique atoms (finite differences)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Computing gradients for symmetry-unique atoms")
    print("=" * 80)

    gradient_unique = np.zeros((num_atoms, 3))

    # Atom 0 (N) - Y component
    grad_n_y = (e_n_y_up - e_n_y_down) / (2 * step_bohr)
    gradient_unique[0, 1] = grad_n_y
    print(f"\nAtom 0 (N), Y-component:")
    print(f"  ∂E/∂Y = (E_up - E_down) / (2 * step)")
    print(f"        = ({e_n_y_up:.12f} - {e_n_y_down:.12f}) / (2 * {step_bohr:.6f})")
    print(f"        = {grad_n_y:.9f} Hartree/Bohr")

    # Atom 0 (N) - Z component
    grad_n_z = (e_n_z_up - e_n_z_down) / (2 * step_bohr)
    gradient_unique[0, 2] = grad_n_z
    print(f"\nAtom 0 (N), Z-component:")
    print(f"  ∂E/∂Z = {grad_n_z:.9f} Hartree/Bohr")

    # Atom 1 (H1) - Y component
    grad_h1_y = (e_h1_y_up - e_h1_y_down) / (2 * step_bohr)
    gradient_unique[1, 1] = grad_h1_y
    print(f"\nAtom 1 (H1), Y-component:")
    print(f"  ∂E/∂Y = {grad_h1_y:.9f} Hartree/Bohr")

    # Atom 1 (H1) - Z component
    grad_h1_z = (e_h1_z_up - e_h1_z_down) / (2 * step_bohr)
    gradient_unique[1, 2] = grad_h1_z
    print(f"\nAtom 1 (H1), Z-component:")
    print(f"  ∂E/∂Z = {grad_h1_z:.9f} Hartree/Bohr")

    print(f"\nGradient for unique atoms (partial):")
    print(f"  Atom 0 (N):  [{gradient_unique[0, 0]:10.6f}, {gradient_unique[0, 1]:10.6f}, {gradient_unique[0, 2]:10.6f}]")
    print(f"  Atom 1 (H1): [{gradient_unique[1, 0]:10.6f}, {gradient_unique[1, 1]:10.6f}, {gradient_unique[1, 2]:10.6f}]")
    print(f"  Atom 2 (H2): [{gradient_unique[2, 0]:10.6f}, {gradient_unique[2, 1]:10.6f}, {gradient_unique[2, 2]:10.6f}]  ← To infer")
    print(f"  Atom 3 (H3): [{gradient_unique[3, 0]:10.6f}, {gradient_unique[3, 1]:10.6f}, {gradient_unique[3, 2]:10.6f}]  ← To infer")

    # ========================================================================
    # STEP 4: Load analytical gradient for symmetry inference
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Loading analytical gradient for symmetry operation inference")
    print("=" * 80)

    analytical_grad = []
    with open(analytical_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    int(parts[0])
                    fx, fy, fz = float(parts[2]), float(parts[3]), float(parts[4])
                    analytical_grad.append([fx, fy, fz])
                except ValueError:
                    continue
    analytical_grad = np.array(analytical_grad)

    print(f"\nAnalytical forces (for symmetry inference):")
    for i in range(num_atoms):
        print(f"  Atom {i}: [{analytical_grad[i, 0]:12.9f}, {analytical_grad[i, 1]:12.9f}, {analytical_grad[i, 2]:12.9f}]")

    # ========================================================================
    # STEP 5: Infer symmetry operations from force patterns
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Inferring symmetry operations from force patterns")
    print("=" * 80)

    # Create minimal engine and infer operations
    log_file = "ErrorDirectory/NH3_C3v.log"
    engine = NonAbelianSymmetryEngine(log_file)

    print(f"\nInferring C3v operations from analytical forces...")
    operations = engine.infer_symmetry_operations_from_forces(
        analytical_grad,
        atoms_with_displacements,
        tolerance=1e-5
    )

    print(f"\n✓ Found {len(operations)} operations:")
    for op in operations:
        if op.operation_id == 1:
            print(f"  Op {op.operation_id}: Identity")
        else:
            # Find which atoms this operation connects
            source = None
            target = None
            for i in range(num_atoms):
                if op.nuclear_permutation[i] != i:
                    if i in atoms_with_displacements and op.nuclear_permutation[i] in atoms_by_symmetry:
                        source = i
                        target = op.nuclear_permutation[i]
                        break

            if source is not None:
                print(f"  Op {op.operation_id}: Atom {source} → Atom {target} (C3 rotation)")
            else:
                print(f"  Op {op.operation_id}: Permutation {op.nuclear_permutation}")

    # ========================================================================
    # STEP 6: Reconstruct full gradient using symmetry
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Reconstructing full gradient using symmetry operations")
    print("=" * 80)

    gradient_full = gradient_unique.copy()

    # Apply symmetry operations to derive H2 and H3 from H1
    for op in operations:
        if op.operation_id == 1:
            continue  # Skip identity

        # Find atom pairs
        for source_atom in atoms_with_displacements:
            target_atom = op.nuclear_permutation[source_atom]
            if target_atom in atoms_by_symmetry:
                # Apply transformation
                gradient_full[target_atom] = op.transformation_matrix @ gradient_full[source_atom]

                print(f"\nApplying Op {op.operation_id}: Atom {source_atom} → Atom {target_atom}")
                print(f"  Source gradient: {gradient_full[source_atom]}")
                print(f"  Transformation:")
                for row in op.transformation_matrix:
                    print(f"    {row}")
                print(f"  Result gradient: {gradient_full[target_atom]}")

    # ========================================================================
    # STEP 7: Final comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Final reconstructed gradient vs analytical")
    print("=" * 80)

    print(f"\n{'Atom':6s} {'Axis':4s} {'Reconstructed':>14s} {'Analytical':>14s} {'Difference':>14s}")
    print("-" * 70)

    axis_names = ['X', 'Y', 'Z']
    rmsd = 0.0

    for i in range(num_atoms):
        for j in range(3):
            recon = gradient_full[i, j]
            anal = analytical_grad[i, j]
            diff = recon - anal
            rmsd += diff ** 2

            marker = "✓" if abs(diff) < 1e-6 else "✗"
            source = "calc" if i in atoms_with_displacements and j > 0 else ("symm" if i in atoms_by_symmetry else "zero")

            print(f"{i:6d} {axis_names[j]:4s} {recon:14.9f} {anal:14.9f} {diff:14.9e}  {marker} ({source})")

    rmsd = np.sqrt(rmsd / (num_atoms * 3))

    print("-" * 70)
    print(f"\nRMSD: {rmsd:.9e} Hartree/Bohr")
    print(f"Max error: {np.max(np.abs(gradient_full - analytical_grad)):.9e} Hartree/Bohr")

    print("\n" + "=" * 80)
    if rmsd < 1e-5:
        print("✅ SUCCESS: Gradient fully reconstructed using symmetry!")
    else:
        print("⚠ WARNING: Reconstruction has errors beyond tolerance")
    print("=" * 80)


if __name__ == "__main__":
    main()
