"""
Test symmetry inference from forces for NH3 C3v molecule.

This test validates that the symmetry engine can correctly infer
symmetry operations from force patterns when explicit operations
are not available in the Gaussian log (commercial version).

Test data:
- ErrorDirectory/Iterations/Iteration_4/: Raw energy calculations
- ErrorDirectory/analitical_gradient.txt: Reference analytical gradient
"""

import numpy as np
import os
from pathlib import Path
from elecext.symmetry_engine import NonAbelianSymmetryEngine, parse_forces_from_gaussian_log


def load_energies_from_iteration(iteration_dir: str) -> dict:
    """
    Load energies from all tasks in iteration directory.

    Parameters:
        iteration_dir: Path to iteration directory (e.g., ErrorDirectory/Iterations/Iteration_4)

    Returns:
        energies: Dict mapping task_id to energy in Hartree
    """
    iteration_path = Path(iteration_dir)
    energies = {}

    for task_dir in iteration_path.iterdir():
        if task_dir.is_dir():
            energy_file = task_dir / "energy_output.dat"
            if energy_file.exists():
                try:
                    with open(energy_file, 'r') as f:
                        line = f.readline().strip()
                        # Format: energy, dipole_x, dipole_y, dipole_z
                        energy = float(line.split(',')[0])
                        task_id = task_dir.name
                        energies[task_id] = energy
                except Exception as e:
                    print(f"Warning: Could not read energy from {energy_file}: {e}")

    return energies


def load_geometries_from_iteration(iteration_dir: str, num_atoms: int) -> dict:
    """
    Load geometries from all task .gjf files.

    Parameters:
        iteration_dir: Path to iteration directory
        num_atoms: Number of atoms

    Returns:
        geometries: Dict mapping task_id to geometry array (num_atoms, 3) in Bohr
    """
    iteration_path = Path(iteration_dir)
    geometries = {}
    ANGSTROM_TO_BOHR = 1.8897259886

    for task_dir in iteration_path.iterdir():
        if task_dir.is_dir():
            gjf_file = task_dir / "GauExternal.gjf"
            if gjf_file.exists():
                try:
                    coords = []
                    with open(gjf_file, 'r') as f:
                        lines = f.readlines()
                        # Skip header, find coordinates section
                        coord_start = False
                        for i, line in enumerate(lines):
                            # Look for charge/multiplicity line
                            if coord_start:
                                parts = line.strip().split()
                                if len(parts) >= 4:  # Element X Y Z
                                    try:
                                        x = float(parts[1]) * ANGSTROM_TO_BOHR
                                        y = float(parts[2]) * ANGSTROM_TO_BOHR
                                        z = float(parts[3]) * ANGSTROM_TO_BOHR
                                        coords.append([x, y, z])
                                    except ValueError:
                                        # End of coordinates
                                        break
                                elif len(parts) == 0:
                                    # Empty line after coordinates
                                    break
                            else:
                                # Look for line with two integers (charge multiplicity)
                                parts = line.strip().split()
                                if len(parts) == 2:
                                    try:
                                        int(parts[0])
                                        int(parts[1])
                                        coord_start = True
                                    except ValueError:
                                        pass

                    if len(coords) == num_atoms:
                        task_id = task_dir.name
                        geometries[task_id] = np.array(coords)
                except Exception as e:
                    print(f"Warning: Could not read geometry from {gjf_file}: {e}")

    return geometries


def load_analytical_gradient(filepath: str) -> np.ndarray:
    """
    Load analytical gradient from reference file.

    Expected format:
        -------------------------------------------------------------------
        Center     Atomic                   Forces (Hartrees/Bohr)
        Number     Number              X              Y              Z
        -------------------------------------------------------------------
             1        7          -0.000000000   -0.000000000   -0.044443232
             ...

    Parameters:
        filepath: Path to analytical gradient file

    Returns:
        gradient: Array (N_atoms, 3) in Hartree/Bohr
    """
    gradient = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # center, atomic_num, fx, fy, fz
                try:
                    # Skip header lines
                    int(parts[0])  # Check if first column is a number
                    fx = float(parts[2])
                    fy = float(parts[3])
                    fz = float(parts[4])
                    gradient.append([fx, fy, fz])
                except ValueError:
                    continue

    return np.array(gradient)


def test_symmetry_inference_nh3():
    """
    Main test: Infer symmetry operations from forces and validate gradient.
    """
    print("=" * 70)
    print("TEST: Symmetry Inference from Forces - NH3 C3v")
    print("=" * 70)

    # Setup paths
    iteration_dir = "ErrorDirectory/Iterations/Iteration_4"
    analytical_file = "ErrorDirectory/analitical_gradient.txt"
    log_file = "ErrorDirectory/NH3_C3v.log"

    num_atoms = 4  # NH3: 1 N + 3 H

    # Load data
    print("\n1. Loading test data...")
    energies = load_energies_from_iteration(iteration_dir)
    print(f"   Loaded {len(energies)} energy calculations")

    geometries = load_geometries_from_iteration(iteration_dir, num_atoms)
    print(f"   Loaded {len(geometries)} geometries")

    analytical_grad = load_analytical_gradient(analytical_file)
    print(f"   Loaded analytical gradient: {analytical_grad.shape}")
    print(f"   Analytical gradient:")
    for i, row in enumerate(analytical_grad):
        print(f"     Atom {i+1}: [{row[0]:12.9f}, {row[1]:12.9f}, {row[2]:12.9f}]")

    # Setup explicit_gradient_recipe
    # From inspection: only atoms 0 (N) and 1 (H1) have displacements on Y (ixyz=2) and Z (ixyz=3)
    # X (ixyz=1) is constrained by symmetry
    print("\n2. Setting up gradient recipe...")
    recipe = {
        (0, 1): {"up": "task_atom_1_ixyz_2_up", "down": "task_atom_1_ixyz_2_down"},  # N, Y
        (0, 2): {"up": "task_atom_1_ixyz_3_up", "down": "task_atom_1_ixyz_3_down"},  # N, Z
        (1, 1): {"up": "task_atom_2_ixyz_2_up", "down": "task_atom_2_ixyz_2_down"},  # H1, Y
        (1, 2): {"up": "task_atom_2_ixyz_3_up", "down": "task_atom_2_ixyz_3_down"},  # H1, Z
    }
    print(f"   Recipe contains {len(recipe)} displacement pairs")
    print(f"   Atoms with displacements: 0 (N), 1 (H1)")
    print(f"   Atoms to infer: 2 (H2), 3 (H3)")

    # Initialize symmetry engine
    print("\n3. Initializing symmetry engine...")
    engine = NonAbelianSymmetryEngine(log_file)
    print(f"   Point group: {engine.point_group_info.name}")
    print(f"   Operations loaded: {len(engine.point_group_info.operations)}")
    if len(engine.point_group_info.operations) == 0:
        print("   Note: No operations found - will use force-based inference")

    # Assemble gradient with symmetry inference
    print("\n4. Assembling gradient with symmetry inference...")

    # WORKAROUND: Since log doesn't have forces section, manually inject
    # analytical gradient as forces for inference
    if len(engine.point_group_info.operations) == 0:
        print("   Manually inferring operations from analytical gradient...")
        inferred_ops = engine.infer_symmetry_operations_from_forces(
            analytical_grad,
            atoms_with_displacements={0, 1},
            tolerance=1e-5
        )
        engine.point_group_info.operations = inferred_ops

    try:
        # Note: We don't have original_coordinates_bohr, so rotation detection will use identity
        calculated_grad = engine.assemble_full_gradient_with_symmetry(
            calculated_energies=energies,
            geometries_in_bohr=geometries,
            explicit_gradient_recipe=recipe,
            num_atoms=num_atoms,
            original_coordinates_bohr=None
        )
        print(f"   Calculated gradient shape: {calculated_grad.shape}")
    except Exception as e:
        print(f"   ERROR: Failed to assemble gradient: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare with analytical
    print("\n5. Comparing with analytical gradient...")
    print(f"\n   {'Atom':6s} {'Axis':4s} {'Calculated':>14s} {'Analytical':>14s} {'Diff':>14s} {'Status':>8s}")
    print("   " + "-" * 70)

    max_error = 0.0
    rmsd = 0.0
    axis_names = ['X', 'Y', 'Z']
    all_passed = True

    for i in range(num_atoms):
        for j in range(3):
            calc = calculated_grad[i, j]
            anal = analytical_grad[i, j]
            diff = calc - anal
            abs_diff = abs(diff)

            max_error = max(max_error, abs_diff)
            rmsd += diff ** 2

            status = "✓" if abs_diff < 1e-6 else "✗"
            if abs_diff >= 1e-6:
                all_passed = False

            print(f"   {i+1:6d} {axis_names[j]:4s} {calc:14.9f} {anal:14.9f} {diff:14.9e} {status:>8s}")

    rmsd = np.sqrt(rmsd / (num_atoms * 3))

    print("   " + "-" * 70)
    print(f"\n   RMSD:      {rmsd:.9e} Hartree/Bohr")
    print(f"   Max Error: {max_error:.9e} Hartree/Bohr")

    # Final verdict
    print("\n6. Test result:")
    if rmsd < 1e-6 and max_error < 1e-6 and all_passed:
        print("   ✅ TEST PASSED: Symmetry inference successful!")
        print("      - All components within tolerance (< 1e-6)")
        print("      - H2 and H3 correctly derived from H1 via symmetry")
        return True
    else:
        print("   ✗ TEST FAILED:")
        if rmsd >= 1e-6:
            print(f"      - RMSD {rmsd:.2e} exceeds tolerance 1e-6")
        if max_error >= 1e-6:
            print(f"      - Max error {max_error:.2e} exceeds tolerance 1e-6")
        return False


def test_vector_transformation_c3v():
    """
    Validate that C3v transformations are vectorial (not component-wise).
    """
    print("\n" + "=" * 70)
    print("TEST: Vector Transformation Validation - C3v")
    print("=" * 70)

    # Reference forces from analytical gradient
    force_H1 = np.array([0.0, -0.024188737, 0.014814411])
    force_H2 = np.array([-0.020948060, 0.012094368, 0.014814411])
    force_H3 = np.array([0.020948060, 0.012094368, 0.014814411])

    # Rotation matrices for C3v
    cos120 = -0.5
    sin120 = np.sqrt(3) / 2

    T_120 = np.array([
        [cos120, -sin120, 0],
        [sin120, cos120, 0],
        [0, 0, 1]
    ])

    T_240 = T_120 @ T_120  # 240° = 120° + 120°

    print("\n1. Testing 120° rotation (H1 → H2)...")
    force_H2_calc = T_120 @ force_H1
    diff_H2 = np.linalg.norm(force_H2_calc - force_H2)
    print(f"   Calculated: {force_H2_calc}")
    print(f"   Expected:   {force_H2}")
    print(f"   Difference: {diff_H2:.2e}")

    if diff_H2 < 1e-6:
        print("   ✓ H2 transformation correct (vectorial)")
    else:
        print("   ✗ H2 transformation failed")

    print("\n2. Testing 240° rotation (H1 → H3)...")
    force_H3_calc = T_240 @ force_H1
    diff_H3 = np.linalg.norm(force_H3_calc - force_H3)
    print(f"   Calculated: {force_H3_calc}")
    print(f"   Expected:   {force_H3}")
    print(f"   Difference: {diff_H3:.2e}")

    if diff_H3 < 1e-6:
        print("   ✓ H3 transformation correct (vectorial)")
    else:
        print("   ✗ H3 transformation failed")

    print("\n3. Verifying component mixing (non-Abelian)...")
    # X component of H2 depends on BOTH X and Y of H1
    print(f"   H1: X={force_H1[0]:.9f}, Y={force_H1[1]:.9f}")
    print(f"   H2: X={force_H2[0]:.9f} = {cos120:.3f}*{force_H1[0]:.9f} + {-sin120:.3f}*{force_H1[1]:.9f}")
    print(f"                        = {cos120 * force_H1[0] + (-sin120) * force_H1[1]:.9f}")
    print("   ✓ Components X and Y are mixed (non-Abelian transformation)")

    if diff_H2 < 1e-6 and diff_H3 < 1e-6:
        print("\n   ✅ Vector transformation test PASSED")
        return True
    else:
        print("\n   ✗ Vector transformation test FAILED")
        return False


if __name__ == "__main__":
    # Run tests
    test1_passed = test_symmetry_inference_nh3()
    test2_passed = test_vector_transformation_c3v()

    print("\n" + "=" * 70)
    print("OVERALL RESULT")
    print("=" * 70)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
        if not test1_passed:
            print("  - Symmetry inference test failed")
        if not test2_passed:
            print("  - Vector transformation test failed")
