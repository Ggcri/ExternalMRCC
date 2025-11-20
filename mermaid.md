# Workflow ExternalMRCC - Integration with Electronic Structure Codes

## Overview Flowchart

```mermaid
flowchart TB
    %% Main components
    Gaussian["<b>Gaussian</b><br/>External Interface"]

    CentralExt["<b>CentralExt</b><br/>Central Dispatcher<br/>━━━━━━━<br/>Receives .EIn (Bohr)<br/>Orchestrates workflow"]

    %% Utilities layer
    subgraph Utils["<b>elecext Utilities</b>"]
        Parse["GauInpParser<br/>parse_scheme_and_formula"]
        Symm["NonAbelianSymmetryEngine<br/>━━━━━━━<br/>• Extract symmetry ops<br/>• Infer from forces (C3v, D6h)<br/>• Vector gradient transform<br/>• Reduce calculations"]
        Combine["combine_gradients<br/>write_output"]
    end

    %% Electronic structure codes
    subgraph Codes["<b>Electronic Structure Codes</b>"]
        GauExt["<b>GauExt</b><br/>━━━━━━━<br/>Input: GauExternal.gjf (Å)<br/>Parse: .fchk, .log<br/>+ Symmetry from log"]

        MRCC["<b>MRCC_ext</b><br/>━━━━━━━<br/>Input: MINP (Bohr XYZ)<br/>Parse: mrcc_output.out<br/>No symmetry handling"]

        Molpro["<b>MolproExt</b><br/>━━━━━━━<br/>Input: qmolpro.com (Bohr)<br/>Parse: .xml gradients<br/>noorient (no symmetry)"]
    end

    %% Data formats
    Input["<b>.EIn File</b><br/>Format: Gaussian External<br/>Units: Bohr<br/>atoms, geom, charge, spin"]

    Output["<b>.EOut File</b><br/>Line 1: E, dipole<br/>Lines 2+: gradients<br/>Units: Hartree/Bohr"]

    Config["<b>Preamble/Ending</b><br/>!scheme, basis,<br/>calc, dens=2"]

    %% Connections
    Gaussian -->|".EIn"| Input
    Input --> CentralExt
    Config -.-> CentralExt

    CentralExt --> Parse
    Parse --> GauExt
    Parse --> MRCC
    Parse --> Molpro

    GauExt <-->|"For parallel<br/>gradients"| Symm

    GauExt --> Combine
    MRCC --> Combine
    Molpro --> Combine

    Combine --> Output
    Output --> CentralExt
    CentralExt -->|".EOut"| Gaussian

    %% Styling
    classDef central fill:#4A90E2,stroke:#2E5C8A,stroke-width:3px,color:#fff
    classDef util fill:#50C878,stroke:#2D7A4A,stroke-width:2px,color:#fff
    classDef code fill:#FF6B6B,stroke:#C44545,stroke-width:2px,color:#fff
    classDef data fill:#FFD93D,stroke:#CCB030,stroke-width:2px,color:#000

    class CentralExt central
    class Parse,Symm,Combine util
    class GauExt,MRCC,Molpro code
    class Input,Output,Config data
    class Gaussian data
```

## Data Objects and File Formats

### Input/Output Files

| File Type | Format | Units | Content |
|-----------|--------|-------|---------|
| **`.EIn`** | Gaussian External | Bohr | Geometry, atoms, charge, spin, opt_flag |
| **`.EOut`** | Gaussian External | Hartree/Bohr | Energy, dipole moment, gradients |
| **Preamble/Ending** | Custom | - | !scheme, basis, calc, dens=2 |

### Intermediate Files by Code

| Code | Input File | Format | Output Parsing |
|------|------------|--------|----------------|
| **Gaussian** | `GauExternal.gjf` | Angstrom | `.fchk`, `.log` files |
| **MRCC** | `MINP` | Bohr (XYZ) | `mrcc_output.out` |
| **Molpro** | `qmolpro.com` | Bohr (noorient) | `.xml` gradients |

## Non-Abelian Symmetry Treatment

The **NonAbelianSymmetryEngine** (used by GauExt for parallel gradients) implements:

1. **Extract Symmetry Operations** from Gaussian log files:
   - Nuclear permutations (atom mapping under symmetry)
   - 3×3 transformation matrices for coordinates/gradients

2. **Fallback: Infer from Force Patterns**:
   - When explicit operations unavailable
   - Test C₃, C₆ rotations for C3v, D3h, D6h point groups
   - RMSD matching < 10⁻⁵

3. **Vector Gradient Transformations**:
   - **grad[target] = T × grad[source]** (full vector, not component-wise)
   - Critical for non-Abelian groups (C3v, D6h, etc.)

4. **Computational Efficiency**:
   - **Example (C3v NH₃)**: 4-6 displacements instead of 12 (50% reduction)
   - **Example (D6h benzene)**: ~75% reduction in gradient calculations

## Key Variables Exchanged

### Python Objects

- **`geom`**: List of geometry strings (e.g., `["C 0.0 0.0 0.0", ...]`)
- **`atoms`**: Integer (number of atoms)
- **`charge`**: Integer (molecular charge)
- **`spin`**: Integer (spin multiplicity)
- **`opt_flag`**: Integer (0=energy only, 1=gradient calculation)
- **`gradient`**: NumPy array `(N_atoms, 3)` in Hartree/Bohr

### Scheme Operations

From Preamble files:
```
!scheme
! 1.0 -1.0 0.5     # Linear coefficients
!end

!formula           # Optional: non-linear combinations
!e = c1*e1 + c2*sqrt(e2) + c3*e3
!end
```

Operations types:
- `('coeff', value)`: Linear coefficient
- `('copy', index, coeff)`: Reuse gradient/energy from previous section
- Custom formulas: Support for `exp`, `log`, `sqrt`, trigonometric functions
