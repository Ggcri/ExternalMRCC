# MRCC External Interface - Complete User Guide

**A comprehensive guide for using the Gaussian External Interface with MRCC for quantum chemistry calculations.**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
   - [MRCC Module File](#1-mrcc-module-file)
   - [Load Module Before Running](#2-load-module-before-running)
   - [SCRATCH Environment Variable](#3-set-scratch-directory-environment-variable)
   - [External Interface Path Variables](#4-set-external-interface-path-environment-variables)
3. [Basic MRCC Calculation](#basic-mrcc-calculation)
   - [Understanding energy_pattern](#understanding-energy_pattern-energy-extraction-from-mrcc-output)
   - [Using Custom Basis Sets](#using-custom-basis-sets-advanced)
4. [Mixed Mode (Analytical + Numerical Gradients)](#mixed-mode-analytical--numerical-gradients)
5. [Scratch Directory Management](#scratch-directory-management)
6. [Resource Specification (MPI and OpenMP)](#resource-specification-mpi-and-openmp)
7. [Step Size Control for Numerical Gradients](#step-size-control-for-numerical-gradients)
8. [Complete Examples](#complete-examples)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

**Minimal example** for running MRCC through Gaussian:

```gaussian
%chk=molecule.chk
%nproc=1
#p External="CentralExt mrcc 16GB READ 4 1 preamble.dat ending.dat" Force

Water molecule optimization

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

**Command breakdown:**
- `mrcc`: Program name
- `16GB`: Total memory available
- `READ`: Use gradient combination scheme from preamble
- `4`: OpenMP threads
- `1`: MPI processes
- `preamble.dat`: MRCC calculation settings
- `ending.dat`: Additional configuration
- `R`: Layer identifier (Real system)
- `Force`: Request gradient calculation

---

## Environment Setup

### 1. MRCC Module File

Create a module file at `/path/to/modulefiles/MRCC/MRCC_external`:

```tcl
#%Module1.0#####################################################################
##
## MRCC External Interface Module
##
proc ModulesHelp { } {
    puts stderr "Loads environment for MRCC external interface"
}

module-whatis "Loads MRCC libraries and dependencies for external interface"

# MRCC library paths
prepend-path LD_LIBRARY_PATH /path/to/MRCC/lib
prepend-path PATH /path/to/MRCC/bin

# LibFabric libraries (required for MPI communication)
# Check MRCC manual for specific library requirements
prepend-path LD_LIBRARY_PATH /path/to/libfabric/lib
prepend-path PATH /path/to/libfabric/bin

# Intel compilers and MKL (adjust based on your system)
module load intel/mkl/latest
module load intel/compiler/latest
module load intel/mpi/latest
```

**Path Placeholders:**
- `/path/to/MRCC/lib`: Location of MRCC shared libraries
- `/path/to/libfabric/lib`: LibFabric installation (see MRCC manual)
- Adjust module names (`intel/mkl/latest`, etc.) to match your cluster

### 2. Load Module Before Running

In your PBS/SLURM job script:

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00

# Load MRCC environment
module load MRCC/MRCC_external

# Load Gaussian (if needed)
module load gaussian/g16

# Set External interface paths
export ELECEXT_PATH=/path/to/ExternalMRCC/Executables
export PATH=$ELECEXT_PATH:$PATH

# Set scratch directory
export SCRATCH=/scratch/$USER/mrcc_jobs
mkdir -p $SCRATCH

# Run Gaussian job
g16 < input.gjf > output.log
```

### 3. Set Scratch Directory Environment Variable

**CRITICAL:** The `SCRATCH` environment variable controls where MRCC stores temporary calculation files.

**Set in your job script:**

```bash
export SCRATCH=/scratch/$USER/mrcc_jobs
mkdir -p $SCRATCH
```

**What SCRATCH controls:**
- Location of MRCC temporary files (`.moints`, `.55`, `.56`, integral files)
- Scratch directories are named: `$SCRATCH/mrcc-{PID}-{SECTION_ID}`
- Example: `$SCRATCH/mrcc-12345-1` for SECTION1 of process 12345

**Default behavior:**
- If `SCRATCH` is **not set**: defaults to `/tmp` (may be slow or limited space)
- If `SCRATCH` **is set**: uses specified directory (recommended: fast local disk)

**Recommendations:**
- Use fast local storage (e.g., `/scratch`, `/local`, SSD)
- Avoid NFS-mounted directories (slow for large integral files)
- Ensure sufficient space (5-50 GB depending on basis set size)

**Verify in job script:**
```bash
echo "SCRATCH directory: $SCRATCH"
ls -lh $SCRATCH/  # Check available space
```

### 4. Set External Interface Path Environment Variables

**REQUIRED:** The External interface needs to locate its executable files. Set these environment variables in your job script:

```bash
# Path to External interface executables directory
export ELECEXT_PATH=/path/to/ExternalMRCC/Executables

# Add executables to PATH for direct invocation
export PATH=$ELECEXT_PATH:$PATH
```

**What these variables do:**

- **`ELECEXT_PATH`**: Points to the directory containing External executables (`CentralExt`, `MRCC_ext`, etc.)
  - Used by `CentralExt` to locate program-specific wrapper scripts
  - Default if not set: directory where `CentralExt` is located

- **`PATH`**: Ensures External executables can be invoked directly
  - Allows Gaussian to find `CentralExt` when calling External interface
  - Required for the `External="CentralExt ..."` command to work

**Example setup:**

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=16

# External interface paths
export ELECEXT_PATH=/home/user/ExternalMRCC/Executables
export PATH=$ELECEXT_PATH:$PATH

# MRCC environment
module load MRCC/MRCC_external

# Scratch directory
export SCRATCH=/scratch/$USER/mrcc_jobs
mkdir -p $SCRATCH

# Run Gaussian
g16 < input.gjf > output.log
```

**Verification:**

```bash
# Check ELECEXT_PATH is set correctly
echo $ELECEXT_PATH
ls $ELECEXT_PATH/CentralExt  # Should show the CentralExt executable

# Check PATH includes executables
which CentralExt  # Should show path to CentralExt
```

**Troubleshooting:**

- **"CentralExt: command not found"** → `PATH` doesn't include `$ELECEXT_PATH`
- **"Cannot find MRCC_ext"** → `ELECEXT_PATH` not set or pointing to wrong directory
- **"Permission denied"** → Executable files need execute permissions (`chmod +x $ELECEXT_PATH/*`)

---

## Basic MRCC Calculation

### Step 1: Create Preamble File (`mrcc_preamble.dat`)

**Single-section example:**

```
!scheme
! 1.0
!end

!SECTION1
energy_pattern=Total CCSD energy [au]:
basis=aug-cc-pVDZ
calc=CCSD
```

**Explanation:**
- `!scheme`: Defines coefficients for energy/gradient combination
- `! 1.0`: Use 100% of SECTION1's result
- `!SECTION1`: Start of calculation section
- `energy_pattern`: Text pattern to extract energy from MRCC output (see below)
- `basis`: Basis set (MRCC syntax)
- `calc`: Method (CCSD, MP2, LCCSD(T), etc.)

---

### Using Environment Variables for Preamble/Ending Files (Advanced)

**For interoperability** with other codes and to avoid copying preamble/ending files across directories, the External interface supports **environment variable expansion** in file paths.

**Syntax:** Use `$VARIABLE_NAME/path/to/file` in the External command line.

**Example:**

```bash
# Set environment variables in job script:
export PMR=/shared/preambles/mrcc
export EMR=/shared/endings/mrcc

# Gaussian input uses environment variables:
#p External="CentralExt mrcc 16GB READ 4 1 $PMR/PR_CCSDF12 $EMR/test_en.dat parall 8" Force
```

**What happens:**
1. External detects `$PMR/PR_CCSDF12` starts with `$`
2. Resolves `$PMR` → `/shared/preambles/mrcc`
3. Final path: `/shared/preambles/mrcc/PR_CCSDF12`
4. Same for `$EMR/test_en.dat` → `/shared/endings/mrcc/test_en.dat`

**Benefits:**
- **Centralized configuration**: One shared directory for all preamble/ending files
- **No file copying**: Multiple jobs reference same files via environment variables
- **Easy version control**: Update shared files without modifying job scripts
- **Interoperability**: Other external codes can use same preamble/ending library

**Error handling:**
- If environment variable not set → Error: `"Environment variable 'PMR' is not set"`
- If resolved path doesn't exist → Error: `"Resolved path does not exist: /shared/preambles/mrcc/PR_CCSDF12"`

**Job script example:**

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=16

# Define shared preamble/ending directories
export PMR=/shared/preambles/mrcc
export EMR=/shared/endings/mrcc

# Load modules
module load MRCC/MRCC_external
module load gaussian/g16

# Run Gaussian with environment variable paths
g16 < input.gjf > output.log
```

---

### Understanding `energy_pattern` (Energy Extraction from MRCC Output)

**What it does:** The `energy_pattern` keyword defines a **text string** that the External interface searches for in the MRCC output file to extract the final energy value.

**Why it's needed:** Different MRCC methods print energy in different formats. The pattern tells the interface exactly which line contains the energy to use.

**IMPORTANT:** The pattern is **NOT a regex** - it's a simple text string matching using case-insensitive comparison. **No need to escape special characters like parentheses!**

**How to find the correct pattern:**

1. **Run MRCC manually** to see the output format:
   ```bash
   # Create a test MRCC input (MINP file)
   cd test_directory
   dmrcc > test_output.txt

   # Search for energy lines
   grep -i "energy" test_output.txt
   ```

2. **Identify the final energy line** in the output. Examples:
   ```
   Total CCSD energy [au]:              -76.242365
   MP2 energy [au]:                     -76.228915
   Total LCCSD(T) energy [au]:          -76.338146
   Total Hartree-Fock energy:           -76.026760
   ```

3. **Copy the text exactly** as it appears **before** the energy number:
   ```
   # For CCSD:
   energy_pattern=Total CCSD energy [au]:

   # For CCSD(T) - NO escape needed (not a regex!):
   energy_pattern=Total CCSD(T) energy [au]:

   # For LCCSD(T):
   energy_pattern=Total LCCSD(T) energy [au]:

   # For MP2:
   energy_pattern=MP2 energy [au]:

   # For Hartree-Fock:
   energy_pattern=Total Hartree-Fock energy:
   ```

**How the matching works (technical details):**
1. Pattern is converted to lowercase internally
2. Each line in MRCC output is checked if it **starts with** the pattern (case-insensitive)
3. First number found after the pattern is extracted as energy
4. If multiple lines match, the **last match** is used (final energy value)

**Important notes:**
- Pattern matching is **case-insensitive** (e.g., `"MP2 ENERGY"` matches `"MP2 energy"`)
- Pattern should match the text **exactly** including spaces
- Special characters like `()`, `[]`, `.` are treated as **literal characters** (no escaping needed!)
- Only the **beginning** of the line is checked (pattern must be at line start after whitespace)

**Common MRCC energy patterns:**

| Method | MRCC Output Line | energy_pattern |
|--------|------------------|----------------|
| HF | `Total Hartree-Fock energy:` | `energy_pattern=Total Hartree-Fock energy:` |
| MP2 | `MP2 energy [au]:` | `energy_pattern=MP2 energy [au]:` |
| CCSD | `Total CCSD energy [au]:` | `energy_pattern=Total CCSD energy [au]:` |
| CCSD(T) | `Total CCSD(T) energy [au]:` | `energy_pattern=Total CCSD(T) energy [au]:` |
| LCCSD(T) | `Total LCCSD(T) energy [au]:` | `energy_pattern=Total LCCSD(T) energy [au]:` |
| CCSDT | `Total CCSDT energy [au]:` | `energy_pattern=Total CCSDT energy [au]:` |

**Testing your pattern:**
```bash
# After running MRCC, verify the pattern extracts energy correctly:
grep -i "Total CCSD energy" MRCC_output.txt
# Should show: Total CCSD energy [au]:              -76.242365
```

**Troubleshooting:**
- **"Cannot find energy pattern"** → Pattern doesn't match output (check spacing, capitalization)
- **Wrong energy value** → Multiple matches found, check if pattern is too generic
- **See also:** [Troubleshooting section](#problem-cannot-find-energy-pattern) for detailed debugging

---

### Step 2: Create Ending File (`mrcc_ending.dat`)

**Minimal ending file:**

```
# Empty or minimal configuration
# For numerical gradient step size control in parallel mode,
# see: Step Size Control for Numerical Gradients section
# Link: #step-size-control-for-numerical-gradients
```

**Note:** Step size keywords are only used when running parallel numerical gradients with the `parall` mode (see [Step Size Control](#step-size-control-for-numerical-gradients) section).

### Step 3: Write Gaussian Input (`water.gjf`)

```gaussian
%chk=water_ccsd.chk
%nproc=1
%mem=16GB
#p External="CentralExt mrcc 16GB READ 4 1 mrcc_preamble.dat mrcc_ending.dat" Force

Water CCSD gradient

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

### Step 4: Run Calculation

```bash
g16 < water.gjf > water.log
```

---

### Using Custom Basis Sets (Advanced)

MRCC supports custom basis set definitions via the `GENBAS` file. The External interface provides two methods for using custom basis sets.

#### Method 1: Global GENBAS File (All Sections)

Place a `GENBAS` file in the working directory. It will be automatically copied to the scratch directory for all MRCC calculations.

**Example:**

```bash
# Create GENBAS file with custom basis definitions
cat > GENBAS << 'EOF'
# Custom basis set for water
O
aug-cc-pVDZ
H
aug-cc-pVDZ
EOF

# Run Gaussian - GENBAS will be used automatically
g16 < water.gjf > water.log
```

**Preamble file:**
```
!SECTION1
basis=custom
calc=CCSD
```

**What happens:**
1. External detects `GENBAS` file in working directory
2. Copies `GENBAS` to MRCC scratch directory before execution
3. MRCC reads basis set definitions from `GENBAS`

#### Method 2: Section-Specific Basis Files (Per-Section Custom Basis)

For multi-section calculations where different sections need different basis sets, create separate basis files named `basis_N` where `N` is the section number (1, 2, 3, etc.).

**Example with two sections:**

Create basis files:
```bash
# Basis for SECTION1
cat > basis_1 << 'EOF'
O
aug-cc-pVDZ
H
aug-cc-pVDZ
EOF

# Basis for SECTION2
cat > basis_2 << 'EOF'
O
cc-pVTZ
H
cc-pVTZ
EOF
```

**Preamble file:**
```
!scheme
! 1.0 1.0
!end

!SECTION1
basis=custom
calc=MP2

!SECTION2
basis=custom
calc=CCSD
```

**What happens:**
1. External processes SECTION1:
   - Detects `basis=custom` in MINP_1 file
   - Looks for `basis_1` in working directory
   - Copies `basis_1` → `GENBAS` in SECTION1 scratch directory
2. External processes SECTION2:
   - Detects `basis=custom` in MINP_2 file
   - Looks for `basis_2` in working directory
   - Copies `basis_2` → `GENBAS` in SECTION2 scratch directory

**Section to filename mapping:**
- `!SECTION1` → looks for `basis_1`
- `!SECTION2` → looks for `basis_2`
- `!SECTION3` → looks for `basis_3`
- Default section → looks for `basis_default`

#### GENBAS File Format

The `GENBAS` file uses MRCC's basis set format. Consult the MRCC manual for detailed syntax.

**Basic structure:**
```
atom_symbol
basis_set_name_or_definition

atom_symbol
basis_set_name_or_definition
```

**Using MRCC built-in basis names:**
```
O
aug-cc-pVDZ
H
aug-cc-pVDZ
```

**Using explicit basis definitions:**
```
O
# explicit basis set definition
# (see MRCC manual for format)

H
# explicit basis set definition
```

#### Error Handling

**Missing basis file:**
```
ERROR: Custom basis file 'basis_1' not found!
```
→ Create the missing `basis_N` file in the working directory where N is the section number (1, 2, 3, etc.)

**Invalid GENBAS format:**
→ Check MRCC output for basis set parsing errors
→ Verify format matches MRCC manual specifications

#### Complete Example: Custom Basis for Composite Method

**Files structure:**
```
working_directory/
├── composite.gjf          # Gaussian input
├── composite_preamble.dat # Preamble with basis=custom
├── ending.dat             # Ending file
├── basis_1                # Custom basis for SECTION1
└── basis_2                # Custom basis for SECTION2
```

**composite_preamble.dat:**
```
!scheme
! 1.0 -1.0
!end

!SECTION1
energy_pattern=Total CCSD energy [au]:
basis=custom
calc=CCSD

!SECTION2
energy_pattern=MP2 energy [au]:
basis=custom
calc=MP2
```

**basis_1:**
```
O
aug-cc-pVTZ
H
aug-cc-pVTZ
```

**basis_2:**
```
O
aug-cc-pVDZ
H
aug-cc-pVDZ
```

**Result:** Composite energy with CCSD/aug-cc-pVTZ - MP2/aug-cc-pVDZ extrapolation using custom basis definitions.

**Important Notes:**
- Each section automatically looks for its corresponding numbered basis file (`basis_1`, `basis_2`, etc.)
- The number in the filename matches the section number from `!SECTION1`, `!SECTION2`, etc.
- If using only a default section (no `!SECTIONN`), name the file `basis_default`
- All basis files must be in the working directory (where you run Gaussian)

---

## Mixed Mode (Analytical + Numerical Gradients)

**Mixed mode combines:**
- **Analytical gradients**: MRCC computes gradient directly (faster, more accurate)
- **Numerical gradients**: External interface uses finite differences (slower, but works for any method)

### When to Use Mixed Mode

✅ **Use mixed mode when:**
- One method supports analytical gradients (e.g., HF, MP2)
- Other methods don't (e.g., high-level coupled cluster)
- Example: `E_total = E_CCSD(T) - E_MP2_CBS + E_MP2_smallbasis`

### How to Activate Mixed Mode

**Key:** Add `dens=2` to sections that support analytical gradients.

**Example Preamble:**

```
!scheme
! 1.0 -1.0 1.0
!end

!SECTION1
energy_pattern=Total LCCSD(T) energy [au]:
basis=aug-cc-pVDZ
calc=LCCSD(T)
# No dens=2 → NUMERICAL gradient (parallel finite differences)

!SECTION2
energy_pattern=MP2 energy [au]:
basis=cc-pwCVTZ
calc=MP2
dens=2          # ← ANALYTICAL gradient (MRCC computes it)

!SECTION3
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2
dens=2          # ← ANALYTICAL gradient
```

**What happens:**
1. External splits preamble into:
   - `SPLIT_PREAMBLES/preamble_analytical.dat` (SECTION2, SECTION3)
   - `SPLIT_PREAMBLES/preamble_numerical.dat` (SECTION1)
2. Runs analytical sections → gets gradients directly from MRCC
3. Runs numerical sections → parallel finite differences
4. Combines: `Grad_total = 1.0×Grad1 - 1.0×Grad2 + 1.0×Grad3`

### Directory Structure

Mixed mode creates separate iteration directories:

```
Iterations/
├── AN_Iteration_1/              # Analytical calculations
│   └── task_analytical_central/
│       ├── Gau-analytical.EIn
│       └── gradient_output.dat  # Direct from MRCC
├── NUM_Iteration_1/             # Numerical calculations
    ├── task_atom_1_ixyz_1_up/
    ├── task_atom_1_ixyz_1_down/
    └── task_central/
```

---

## Scratch Directory Management

### Why Scratch Matters

MRCC calculations generate large temporary files:
- MO coefficients (`.moints`)
- Two-electron integrals (`.55`, `.56`)
- Checkpoint files

**Reusing scratch** between related calculations saves time.

### Automatic Scratch Naming

Each MRCC section gets a unique scratch directory:

```
$SCRATCH/mrcc-{process_id}-{section_id}
```

**Example:**
- Process ID: `12345`
- SECTION1: `$SCRATCH/mrcc-12345-1`
- SECTION2: `$SCRATCH/mrcc-12345-2`

### Copying Scratch Between Sections (`!cpscr`)

**Use case:** MP2 calculation followed by CCSD that reuses MP2 integrals.

**Preamble:**

```
!scheme
! 1.0 1.0
!end

!SECTION1
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2

!SECTION2
!cpscr 1        # ← Copy scratch from SECTION1 before starting
energy_pattern=CCSD energy [au]:
basis=aug-cc-pVDZ
calc=CCSD
```

**What happens:**
1. SECTION1 runs → creates `mrcc-12345-1/` with MO integrals
2. Before SECTION2 starts: **copies** `mrcc-12345-1/` → `mrcc-12345-2/`
3. SECTION2 reuses integrals → skips SCF/integral generation
4. Both scratches preserved until all dependent calculations finish

### Scratch Cleanup Rules

**Why scratch preservation matters:**

The External interface uses an intelligent scratch management system because:
- MRCC calculations can be very expensive (hours or days of computation)
- Later sections may need integrals/orbitals from earlier sections (via `!cpscr`)
- Deleting scratch too early would force expensive recalculation
- Keeping scratch too long wastes disk space

**The solution:** Dependency-based cleanup tracks which sections need which scratch directories and deletes them only when safe.

**Example dependency tree:**

```
Preamble configuration:
!SECTION1: calc=MP2
!SECTION2: !cpscr 1, calc=CCSD    # Needs SECTION1 scratch
!SECTION3: !cpscr 1, calc=CCSD(T)  # Needs SECTION1 scratch
!SECTION4: calc=HF                 # Independent, no !cpscr

Dependency map built by External:
SECTION2 → depends on SECTION1
SECTION3 → depends on SECTION1
SECTION4 → no dependencies
```

**Automatic cleanup order:**
1. **SECTION1 finishes** → **preserved** (SECTION2 and SECTION3 still need its integrals)
2. **SECTION2 finishes** → **deleted immediately** (no other sections depend on it)
3. **SECTION3 finishes** → **deleted immediately** (no other sections depend on it)
4. **SECTION4 finishes** → **deleted immediately** (was never needed by anyone)
5. **SECTION1 deleted** → All dependents (SECTION2, SECTION3) are done, safe to delete

---

## Resource Specification (MPI and OpenMP)

### IMPORTANT: Per-Process Resources

⚠️ **Critical:** Resources in the command line are **per MPI process**, not total!

**Command:**
```
CentralExt mrcc 16GB READ 4 2 preamble.dat ending.dat R
                ^^^^      ^ ^
                memory    │ └─ MPI processes
                          └─── OpenMP threads per process
```

**Actual allocation:**
- Total memory: `16GB / 2 MPI = 8GB per process`
- Total threads: `4 OpenMP × 2 MPI = 8 total threads`

### Command Line Syntax

**Standard format:**

```bash
CentralExt mrcc <mem> <gradmode> <omp> <mpi> <preamble> <ending> <layer> <input> <output>
```

**Parameters:**
- `<mem>`: Total memory (e.g., `16GB`, `120GB`)
- `<gradmode>`: `READ` (use scheme), `SCAN` (zero gradient), or filename
- `<omp>`: OpenMP threads **per MPI process**
- `<mpi>`: Number of MPI processes
- `<preamble>`: Preamble file path
- `<ending>`: Ending file path
- `<layer>`: Usually `R` (real system) or `H`/`M`/`L` for ONIOM
- `<input>`: Gaussian `.EIn` file
- `<output>`: Output `.EOut` file

### Gaussian Input

```gaussian
%nproc=1                    # Gaussian's own parallelization (usually 1 for external)
%mem=16GB                   # Gaussian memory (separate from MRCC)
#p External="CentralExt mrcc 16GB READ 4 2 preamble.dat ending.dat" Force
```

### Activating MPI in MRCC

**Method 1: Add `mpi` keyword in preamble (case-insensitive)**

```
!SECTION1
energy_pattern=Total CCSD energy [au]:
basis=aug-cc-pVDZ
calc=CCSD
mpi             # ← Triggers MPI mode (keyword without !)
```

**What happens:**
- External detects `mpi` keyword (case-insensitive, works anywhere in line)
- Replaces `mpi` with `mpitasks=2` (using MPI count from command line)
- Writes to MINP file: `mpitasks=2`
- Calculates per-task memory: `mem=8192MB` (16GB / 2 MPI tasks)

**IMPORTANT:** The keyword is `mpi` (lowercase, no `!` prefix). It gets replaced with the actual `mpitasks=N` directive automatically.

**Method 2: Explicit `mpitasks` in preamble**

```
!SECTION1
mpitasks=4      # ← Explicit MPI tasks (overrides command line)
calc=CCSD
```

**Selective MPI usage (different sections, different parallelization):**

You can enable MPI for some sections but not others:

```
!scheme
! 1.0 1.0
!end

!SECTION1
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2
mpi             # ← MPI enabled for this section

!SECTION2
energy_pattern=Total CCSD energy [au]:
basis=aug-cc-pVDZ
calc=CCSD
# No 'mpi' keyword → OpenMP-only for this section
```

**Result:**
- SECTION1 (MP2): Uses 2 MPI × 4 OpenMP = 8 total threads, memory = 8GB per MPI task
- SECTION2 (CCSD): Uses 1 MPI × 4 OpenMP = 4 total threads, memory = 16GB (full allocation)

### Memory Allocation Rules

**Without MPI** (`mpi_procs=1`):
```
MINP file:
mem=16384MB     # Full memory for OpenMP-only
```

**With MPI** (`mpi_procs=2`):
```
MINP file:
mpitasks=2
mem=8192MB      # Per-task memory (16GB / 2)
```

**Special case for MP2:**
```
MP2 calculations use total memory:
mem=16384MB     # Even with MPI
```

### Environment Variables

**OpenMP threads** (automatically set):

```bash
# With MPI=2, OMP=4:
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# OpenMP-only (MPI=1, OMP=8):
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Parallel Numerical Gradients

**Add `parall <nthreads>` for parallel finite differences:**

```gaussian
#p External="CentralExt mrcc 16GB READ 4 2 preamble.dat ending.dat parall 2" Force
                                                                       ^^^^^^^^
                                                                       2 parallel workers
```

**What happens:**
- 2 **independent** MRCC jobs run simultaneously
- Each MRCC job uses: 4 OpenMP × 2 MPI = 8 threads
- Total system load: 2 workers × 8 threads = **16 threads**

**Running parallel gradients:**

```gaussian
# Parallel numerical gradients with 2 workers:
External="... parall 2"
```

---

## Step Size Control for Numerical Gradients

When using `parall` mode, the External interface generates numerical gradients using finite differences along Cartesian displacements. The step size (displacement magnitude) can be controlled via Gaussian's frequency calculation keywords.

### How Step Size Control Works

The External generates displacements by running a **fake Gaussian frequency calculation** (`freq=num`) internally. The step size is controlled by Gaussian's `freq` keyword options.

### Default Behavior

**Without customization:**
- Gaussian uses default step size (~0.01 Bohr for most coordinates)
- Route: `#p freq=num geom=gic uff iop(1/33=2)`
- Works well for most organic molecules

### Customizing Step Size

**Add `!fakekey` section to `ending.dat`:**

```
!fakekey
!freq=step=20
```

**Common options:**

| Keyword | Step Size | Use Case |
|---------|-----------|----------|
| `freq=step=5` | ~0.005 Bohr | Very tight convergence, small molecules |
| `freq=step=10` | ~0.01 Bohr | **Default** (Gaussian standard) |
| `freq=step=20` | ~0.02 Bohr | Faster calculations, larger molecules |
| `freq=step=50` | ~0.05 Bohr | Initial optimization far from minimum |

**Other useful keywords:**

```
!fakekey
!freq=step=10
!nosymm        # Disable symmetry (full Cartesian displacements)
!scf=xqc       # More robust SCF for fake freq calculation
```

### Complete Example

**File:** `ending.dat`

```
# Larger step size for faster gradient calculation
!fakekey
!freq=step=20
!nosymm
```

**What happens:**
1. External runs fake frequency calculation with `freq=step=20 nosymm`
2. Gaussian generates displacements with ~0.02 Bohr step size
3. External extracts displacement geometries
4. Parallel energy calculations run for each displaced geometry
5. Finite difference gradient assembled from energies


---

## Complete Examples

### Example 1: Single CCSD(T) Gradient

**File:** `water_ccsd.gjf`

```gaussian
%chk=water_ccsd.chk
%nproc=1
%mem=16GB
#p External="CentralExt mrcc 16GB READ 8 1 ccsd_preamble.dat ending.dat" Force

Water CCSD(T) gradient optimization

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

**File:** `ccsd_preamble.dat`

```
!scheme
! 1.0
!end

!SECTION1
energy_pattern=Total CCSD\(T\) energy [au]:
basis=aug-cc-pVDZ
calc=CCSD(T)
```

**File:** `ending.dat`

```
# Optional: customize step size for numerical gradients
!fakekey
!freq=step=10
```

**Run:**

```bash
g16 < water_ccsd.gjf > water_ccsd.log
```

---

### Example 2: Composite Method (Mixed Mode)

**File:** `composite.gjf`

```gaussian
%chk=water_composite.chk
%nproc=1
#p External="CentralExt mrcc 120GB READ 16 4 composite_preamble.dat ending.dat parall 2" Opt

Composite CBS extrapolation with mixed gradients

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

**File:** `composite_preamble.dat`

```
!scheme
! 1.0 -1.0 1.0
!formula
!e=c1*e1 + c2*e2 + c3*e3
!end

!SECTION1
energy_pattern=Total LCCSD\(T\) energy [au]:
basis=aug-cc-pVDZ
calc=LCCSD(T)
# Numerical gradient (no dens=2)

!SECTION2
energy_pattern=MP2 energy [au]:
basis=cc-pV5Z
calc=MP2
dens=2          # Analytical gradient

!SECTION3
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2
dens=2          # Analytical gradient
```

**File:** `ending.dat`

```
# Optional: customize step size for numerical gradients
!fakekey
!freq=step=20
```

**What happens:**
1. SECTION1 (LCCSD(T)): Numerical gradient via parallel finite differences (2 workers)
2. SECTION2 (MP2/CBS): Analytical gradient from MRCC
3. SECTION3 (MP2/small): Analytical gradient from MRCC
4. Combined: `Grad = Grad1 - Grad2 + Grad3`

---

### Example 3: Scratch Reuse for CCSD

**File:** `ccsd_reuse.gjf`

```gaussian
%chk=water_ccsd.chk
%nproc=1
#p External="CentralExt mrcc 32GB READ 8 2 ccsd_reuse_preamble.dat ending.dat" Force

CCSD using MP2 integrals

0 1
O     0.000000    0.000000    0.117790
H     0.000000    0.756950   -0.471160
H     0.000000   -0.756950   -0.471160
```

**File:** `ccsd_reuse_preamble.dat`

```
!scheme
! 0.0 1.0
!end

!SECTION1
energy_pattern=MP2 energy [au]:
basis=aug-cc-pVDZ
calc=MP2
mpi             # Use MPI for integrals

!SECTION2
!cpscr 1        # ← Reuse SECTION1 scratch
energy_pattern=CCSD energy [au]:
basis=aug-cc-pVDZ
calc=CCSD
mpi
```

**Benefits:**
- SECTION1 computes integrals (cheap MP2)
- SECTION2 skips integral generation (expensive CCSD)
- Total time: ~30% faster than standalone CCSD

---

## Troubleshooting

### Problem: "MRCC not found in PATH"

**Solution:** Load MRCC module before running Gaussian

```bash
module load MRCC/MRCC_external
which dmrcc    # Should show MRCC binary path
```

---

### Problem: "Out of memory" during MPI calculation

**Cause:** Memory is split between MPI processes

**Solution:** Either reduce MPI processes or increase total memory

```
# Before (fails):
CentralExt mrcc 16GB READ 4 8 ...
                ^^^^         ^
                16GB / 8 = 2GB per process (too small!)

# After (works):
CentralExt mrcc 64GB READ 4 8 ...
                ^^^^         ^
                64GB / 8 = 8GB per process
```

---

### Problem: Scratch directory fills up disk

**Cause:** Large basis set calculations generate huge integral files

**Solutions:**

1. **Set SCRATCH to fast local disk:**
   ```bash
   export SCRATCH=/scratch/$USER
   ```

2. **Enable scratch cleanup** (automatic for most calculations)

3. **Check scratch size:**
   ```bash
   du -sh $SCRATCH/mrcc-*
   ```

---

### Problem: Numerical gradients don't converge

**Symptoms:**
- Optimization oscillates
- Gradient RMS doesn't decrease

**Solutions:**

1. **Increase step size for less noisy gradients:**
   ```
   !fakekey
   !freq=step=20
   ```

2. **Check MRCC energy convergence:**
   - Make sure `itol=` is tight enough in MRCC input
   - Typical: `itol=10` for 10⁻¹⁰ Hartree

3. **Increase parallel workers (if system resources allow):**
   ```
   External="... parall 4"  # Use 4 parallel workers instead of 2
   ```

---

### Problem: "Cannot find energy pattern"

**Cause:** `energy_pattern` regex doesn't match MRCC output

**Solution:** Check MRCC output for exact energy line

```bash
# Run MRCC manually:
cd $SCRATCH/mrcc-test
dmrcc > test.out

# Find energy line:
grep -i "energy" test.out
```

**Example patterns:**
```
# CCSD:
energy_pattern=Total CCSD energy [au]:

# CCSD(T):
energy_pattern=Total CCSD(T) energy [au]:

# MP2:
energy_pattern=MP2 energy [au]:

# HF:
energy_pattern=Total Hartree-Fock energy:
```

---

## Advanced Tips

### Tip 1: Monitor Scratch Usage During Calculation

```bash
# In another terminal:
watch -n 5 'du -sh $SCRATCH/mrcc-*'
```

Shows real-time scratch size every 5 seconds.

---

### Tip 2: Reuse Checkpoint Files Across Jobs

```bash
# After optimization:
formchk water_opt.chk water_opt.fchk

# Start next job from converged geometry:
%oldchk=water_opt.chk
#p External="..." Freq Geom=AllCheck
```

---

### Tip 3: Test Preamble Before Full Calculation

```bash
# Quick test with minimal basis:
!SECTION1
basis=sto-3g     # Fast for testing
calc=HF

# Then switch to production basis:
basis=aug-cc-pVDZ
calc=CCSD(T)
```

---


**Key Takeaways:**

1. Resources are **per MPI process**, not total
2. Add `dens=2` for analytical gradients (mixed mode)
3. Use `!cpscr N` to reuse scratch from section N
4. Default Gaussian step size (`freq=step=10`) works for most cases
5. Always load MRCC module before running calculations

For more examples, see the `Examples/` directory in this repository.
