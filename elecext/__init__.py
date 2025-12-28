# Helper utilities shared by external interfaces
import numpy as np
import os
import re
import ast
import operator
import sys


class Logger:
    """Simple file logger used by external executables."""

    def __init__(self, log_file):
        # Always use absolute path to ensure log file is accessible regardless
        # of working directory changes during execution
        self.log_file = os.path.abspath(log_file)
        # In test mode we keep any existing log so that tests can provide
        # pre-populated files.  Otherwise truncate the log at startup.
        if os.environ.get("EXT_TEST_MODE") != "1":
            # Ensure the directory for the log file exists
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_file, "a") as f:
                pass

    def log(self, *args):
        """Append a generic message to the log file."""
        message = " ".join(str(a) for a in args)
        try:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        except (OSError, IOError) as e:
            # If we can't write to the log file, try to create the directory first
            log_dir = os.path.dirname(self.log_file)
            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    with open(self.log_file, "a") as f:
                        f.write(message + "\n")
                except (OSError, IOError):
                    # If we still can't write, print to stderr as fallback
                    print(f"Logger error: Cannot write to {self.log_file}: {e}", file=sys.stderr)
                    print(f"Log message: {message}", file=sys.stderr)
            else:
                print(f"Logger error: Cannot write to {self.log_file}: {e}", file=sys.stderr)
                print(f"Log message: {message}", file=sys.stderr)

    def header(self, program, nprocs, mem):
        """Write header information about the run."""
        self.log(f"PROGRAM: {program}")
        self.log(f"NPROCS: {nprocs}")
        self.log(f"MEMORY: {mem}")

    def log_energy(self, energy):
        self.log(f"ENERGY: {energy:.12f}")

    def log_gradients(self, iteration, gradients, combined=None):
        self.log(f"ITERATION {iteration}")
        for idx, grad in enumerate(gradients):
            self.log(f"GRADIENT {idx + 1}")
            for line in grad:
                self.log(" ".join(f"{x:20.12f}" for x in line))
        if combined is not None:
            self.log("COMBINED")
            for line in combined:
                self.log(" ".join(f"{x:20.12f}" for x in line))

def parse_coefficients(preamble_file, program="gaussian"):
    """Parse coefficients from a preamble file.

    Parameters
    ----------
    preamble_file : str
        Path to the preamble file.
    program : str, optional
        Name of the electronic structure code. It determines the comment
        character used to mark coefficient lines. ``orca`` uses ``#`` while
        Molpro, MRCC and Gaussian use ``!``.

    Returns
    -------
    list
        List of floating point coefficients. If no coefficient is found a
        default ``[1.0]`` list is returned.
    """

    if program.lower() == "orca":
        comment_char = "#"
    else:
        comment_char = "!"

    coefficients = []
    with open(preamble_file, "r") as file:
        in_scheme = False
        for line in file:
            line = line.strip()
            if "scheme" in line.lower():
                in_scheme = True
                continue

            if in_scheme:
                if line.startswith(comment_char):
                    value_str = line[1:].strip()
                    try:
                        coefficients.append(float(value_str))
                    except ValueError:
                        try:
                            coefficients.extend(float(v) for v in value_str.split())
                        except ValueError:
                            pass
                if not line or "end" in line.lower():
                    break

    if not coefficients:
        coefficients = [1.0]

    return coefficients


def combine_gradients(coefficients, gradients):
    """Linearly combine multiple gradients using the given coefficients."""
    gradients = np.asarray(gradients, dtype=float)
    combined = np.zeros_like(gradients[0], dtype=float)
    for coef, grad in zip(coefficients, gradients):
        combined += coef * grad
    return combined


def GauInpParser(gau_input, spin_offset=0, scale=1.0):
    """Parse a Gaussian-style geometry input file."""
    with open(gau_input, 'r') as file:
        content = file.readlines()

    atoms = int(content[0].split()[0])
    charge = int(content[0].split()[2])
    spin = int(content[0].split()[3]) + spin_offset
    opt_flag = int(content[0].split()[1])

    atomic_symbols = {
        '1': 'H', '2': 'He', '3': 'Li', '4': 'Be', '5': 'B', '6': 'C',
        '7': 'N', '8': 'O', '9': 'F', '10': 'Ne', '11': 'Na', '12': 'Mg',
        '13': 'Al', '14': 'Si', '15': 'P', '16': 'S', '17': 'Cl',
        '18': 'Ar', '19': 'K', '20': 'Ca'
    }

    geom = [
        ' '.join([atomic_symbols[line.split()[0]]] +
                 [str(float(coord) * scale) for coord in line.split()[1:4]])
        for line in content[1:atoms+1]
    ]

    return geom, atoms, spin, charge, opt_flag


def write_output(gau_output, energy, dipole_moment=None, gradient=None):
    """Write energy, dipole moment and optional gradient to Gaussian output."""
    if dipole_moment is None:
        dipole_moment = [0.0, 0.0, 0.0]
    line = "{:20.12f},{:20.12f},{:20.12f},{:20.12f}\n".format(energy, *dipole_moment)
    with open(gau_output, 'w') as file:
        file.write(line)
        if gradient is not None:
            for grad in gradient:
                file.write(''.join(f"{g:20.12f}" for g in grad) + "\n")


def parse_scheme_and_formula(preamble_file, program='mrcc'):
    """
    Parse coefficients and formula from a preamble file marked within a !scheme ... !end block.
    Supports both simple coefficients and copy[index,coeff] operations like MolproExt_project.
    Also extracts custom formula if present after !formula marker.
    
    Args:
        preamble_file (str): Path to the preamble file.
        program (str): Program type ('mrcc', 'molpro' use '!', others default to '#').
        
    Returns:
        tuple: (parsed_operations, formula)
            - parsed_operations (list): List of operation tuples: ('coeff', value) or ('copy', source_index, coefficient)
            - formula (str or None): Custom formula string if found, None otherwise
    """
    parsed_operations = []
    formula = None
    comment_char = '!' if program.lower() in ['mrcc', 'molpro', 'gaussian'] else '#'
    in_scheme_section = False
    
    # Regex for finding "copy[index,coefficient]"
    # Captures the index (integer) and coefficient (float)
    copy_regex = re.compile(r"copy\[\s*(\d+)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\]")

    try:
        with open(preamble_file, 'r') as file:
            print(f"Parsing scheme and formula from {preamble_file} (comment='{comment_char}')")
            lines = file.readlines()
            for idx, line in enumerate(lines):
                line_strip = line.strip()

                # Detect start of scheme block (case-insensitive)
                if '!scheme' in line_strip.lower():
                    if in_scheme_section:
                         print(f"  Warning: Nested '!scheme' detected at line {idx+1}. Ignoring inner start.")
                    else:
                         in_scheme_section = True
                         print(f"  Entered !scheme section at line {idx+1}.")
                    continue

                # Detect end of scheme block (case-insensitive)
                if '!end' in line_strip.lower():
                    if not in_scheme_section:
                         print(f"  Warning: Found '!end' outside of a !scheme block at line {idx+1}. Ignoring.")
                    else:
                         in_scheme_section = False
                         print(f"  Exited !scheme section at line {idx+1}.")
                    break # Exit scheme processing

                # Process lines within the scheme section
                if in_scheme_section:
                    # Check for formula marker
                    if line_strip.lower() == '!formula':
                        print(f"  Found !formula marker at line {idx+1}")
                        # Look for formula on the next non-empty, non-comment line
                        for i in range(idx + 1, len(lines)):
                            formula_line_content = lines[i].strip()
                            if not formula_line_content or formula_line_content.startswith('#'):
                                continue
                            if formula_line_content.startswith('!') and not formula_line_content.lower().startswith('!end'):
                                # This is our formula line
                                formula = formula_line_content[1:].strip()  # Remove '!' prefix
                                print(f"    Found formula at line {i+1}: {formula}")
                                # Skip the formula line in subsequent processing by marking it as processed
                                lines[i] = '# FORMULA_PROCESSED\n'
                                break
                        continue

                    # Check if the stripped line starts with the comment character (for coefficients)
                    if line_strip.startswith(comment_char):
                        # Extract text *after* the comment character
                        value_part = line_strip[len(comment_char):].strip()
                        if not value_part: # Skip empty comments like '!'
                            continue

                        # Parse coefficients and copy operations
                        parts = value_part.split()
                        for part_str in parts:
                            match = copy_regex.fullmatch(part_str)
                            if match:
                                source_index = int(match.group(1))
                                coeff_value = float(match.group(2))
                                if source_index <= 0:
                                    raise ValueError(f"Copy index must be positive: {part_str}")
                                parsed_operations.append(('copy', source_index, coeff_value))
                                print(f"    Line {idx+1}: Found copy operation: copy[{source_index}, {coeff_value}]")
                            else:
                                try:
                                    coeff_value = float(part_str)
                                    parsed_operations.append(('coeff', coeff_value))
                                    print(f"    Line {idx+1}: Found coefficient: {coeff_value}")
                                except ValueError:
                                    # Check if it's the start of an inline comment and stop parsing this line
                                    if part_str.startswith('#') or part_str.startswith('!'):
                                        break
                                    raise ValueError(f"Invalid part in scheme: '{part_str}'. Must be a number or 'copy[idx,coeff]'.")

    except FileNotFoundError:
        print(f"Warning: Preamble file '{preamble_file}' not found for scheme parsing.")
    except Exception as e:
        print(f"Error reading or parsing scheme from '{preamble_file}': {e}")

    # If no operations were found within a valid !scheme block, return default
    if not parsed_operations:
         print(f"Info: No operations found in scheme. Using default coefficient [1.0].")
         return [('coeff', 1.0)], formula
    else:
        print(f"Finished parsing scheme: {parsed_operations}")
        if formula:
            print(f"Found formula: {formula}")
        return parsed_operations, formula


def evaluate_custom_formula(formula, energies, coefficients):
    """
    Safely evaluate a custom formula for combining energies and gradients.
    
    Args:
        formula (str): Formula string (e.g., "e=exp(c1*e1)+log(c2*e2)+c3*e3")
        energies (list): List of energy values or gradient arrays
        coefficients (list): List of coefficient values
        
    Returns:
        float or numpy.ndarray: Evaluated result (scalar for energies, array for gradients)
    """
    
    # Define allowed functions
    safe_functions = {
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'abs': np.abs,
        'pow': np.power,
        'pi': np.pi,
        'e': np.e
    }
    
    # Define allowed operators
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    class FormulaEvaluator(ast.NodeVisitor):
        def __init__(self, variables):
            self.variables = variables
            
        def visit_BinOp(self, node):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op = safe_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
            
        def visit_UnaryOp(self, node):
            operand = self.visit(node.operand)
            op = safe_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)
            
        def visit_Call(self, node):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in safe_functions:
                raise ValueError(f"Unsupported function: {func_name}")
            
            args = [self.visit(arg) for arg in node.args]
            return safe_functions[func_name](*args)
            
        def visit_Name(self, node):
            if node.id in self.variables:
                return self.variables[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
                
        def visit_Constant(self, node):
            return node.value
            
        def visit_Num(self, node):  # For Python < 3.8 compatibility
            return node.n
            
        def generic_visit(self, node):
            raise ValueError(f"Unsupported AST node: {type(node).__name__}")
    
    try:
        # Parse the formula (remove 'e=' prefix if present)
        if '=' in formula:
            formula = formula.split('=', 1)[1].strip()
        
        # Create variable mapping
        variables = {}
        
        # Add energy variables (e1, e2, e3, ...)
        for i, energy in enumerate(energies, 1):
            variables[f'e{i}'] = energy
            
        # Add coefficient variables (c1, c2, c3, ...)
        for i, coeff in enumerate(coefficients, 1):
            variables[f'c{i}'] = coeff
            
        # Parse and evaluate the formula
        tree = ast.parse(formula, mode='eval')
        evaluator = FormulaEvaluator(variables)
        result = evaluator.visit(tree.body)
        
        print(f"Formula evaluation successful: {formula}")
        print(f"Variables used: {list(variables.keys())}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: Formula evaluation failed: {e}")
        print(f"Formula: {formula}")
        print(f"Available variables: e1-e{len(energies)}, c1-c{len(coefficients)}")
        raise ValueError(f"Formula evaluation error: {e}")


def resolve_path_with_env(path):
    """
    Resolve file paths that may contain environment variables.
    
    If the path starts with '$', treat it as an environment variable reference
    and resolve it to the actual path.
    
    Args:
        path (str): File path that may start with '$VAR_NAME'
        
    Returns:
        str: Resolved absolute path
        
    Raises:
        ValueError: If environment variable is not set or resolved path doesn't exist
    """
    import os
    
    if not path.startswith('$'):
        # Regular path, check if it exists as-is first
        if os.path.exists(path):
            return os.path.abspath(path)
        # If not found as relative, try as absolute
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path
        # Return the absolute path anyway (let the caller handle FileNotFoundError)
        return abs_path
    
    # Extract environment variable name (everything after '$' until first '/')
    if '/' in path:
        env_var, rest_of_path = path[1:].split('/', 1)
        rest_of_path = '/' + rest_of_path
    else:
        env_var = path[1:]
        rest_of_path = ''
    
    # Get environment variable value
    env_value = os.environ.get(env_var)
    if env_value is None:
        raise ValueError(f"Environment variable '{env_var}' is not set for path: {path}")
    
    # Construct the resolved path
    resolved_path = os.path.join(env_value, rest_of_path.lstrip('/'))
    resolved_path = os.path.abspath(resolved_path)
    
    # Verify the resolved path exists
    if not os.path.exists(resolved_path):
        raise ValueError(f"Resolved path does not exist: {resolved_path} (from ${env_var}{rest_of_path})")
    
    return resolved_path


def validate_formula_syntax(formula, num_energies, num_coefficients):
    """
    Validate formula syntax before evaluation.

    Args:
        formula (str): Formula string to validate
        num_energies (int): Number of energy values available
        num_coefficients (int): Number of coefficients available

    Returns:
        bool: True if valid, raises ValueError if invalid
    """

    try:
        # Remove 'e=' prefix if present
        if '=' in formula:
            formula = formula.split('=', 1)[1].strip()

        # Parse the formula
        tree = ast.parse(formula, mode='eval')

        # Check for valid variable names
        allowed_functions = ['exp', 'log', 'log10', 'sqrt', 'sin', 'cos', 'tan', 'abs', 'pow']
        allowed_constants = ['pi', 'e']

        class VariableChecker(ast.NodeVisitor):
            def __init__(self, max_e, max_c):
                self.max_e = max_e
                self.max_c = max_c
                self.variables_used = set()

            def visit_Name(self, node):
                var_name = node.id
                self.variables_used.add(var_name)

                if var_name.startswith('e') and len(var_name) > 1:
                    try:
                        idx = int(var_name[1:])
                        if idx < 1 or idx > self.max_e:
                            raise ValueError(f"Energy variable {var_name} out of range (1-{self.max_e})")
                    except ValueError as e:
                        if "invalid literal" in str(e):
                            raise ValueError(f"Invalid energy variable name: {var_name}")
                        raise
                elif var_name.startswith('c') and len(var_name) > 1:
                    try:
                        idx = int(var_name[1:])
                        if idx < 1 or idx > self.max_c:
                            raise ValueError(f"Coefficient variable {var_name} out of range (1-{self.max_c})")
                    except ValueError as e:
                        if "invalid literal" in str(e):
                            raise ValueError(f"Invalid coefficient variable name: {var_name}")
                        raise
                elif var_name not in allowed_constants:  # Allow mathematical constants
                    # Check if it's being used as a function (handled in visit_Call)
                    # For standalone names, only allow known constants
                    if var_name not in allowed_functions:
                        raise ValueError(f"Unknown variable: {var_name}")

                self.generic_visit(node)

            def visit_Call(self, node):
                # Function calls are allowed for certain mathematical functions
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in allowed_functions:
                        raise ValueError(f"Unknown function: {func_name}")

                # Visit arguments
                for arg in node.args:
                    self.visit(arg)

        checker = VariableChecker(num_energies, num_coefficients)
        checker.visit(tree)

        print(f"Formula validation successful. Variables used: {sorted(checker.variables_used)}")
        return True

    except SyntaxError as e:
        raise ValueError(f"Formula syntax error: {e}")
    except Exception as e:
        raise ValueError(f"Formula validation error: {e}")


def analyze_sections_for_gradient_mode(preamble_file, program='mrcc'):
    """
    Analyze preamble file sections to classify them as analytical or numerical gradient modes.
    Detects 'dens=2' keyword to identify sections that support analytical gradients.

    Args:
        preamble_file (str): Path to the preamble file
        program (str): Program type ('mrcc', 'molpro' use '!', others use '#')

    Returns:
        dict: Section metadata with structure:
            {
                'default': {'dens': 0, 'has_analytical': False, 'content': [...]},
                '!SECTION1': {'dens': 2, 'has_analytical': True, 'content': [...]},
                '!SECTION2': {'dens': 0, 'has_analytical': False, 'content': [...]},
            }
    """
    section_metadata = {}
    comment_char = '!' if program.lower() in ['mrcc', 'molpro', 'gaussian'] else '#'
    current_section_id = "default"
    section_metadata[current_section_id] = {
        'dens': 0,  # Default value
        'has_analytical': False,
        'content': []
    }

    in_scheme_block = False

    try:
        with open(preamble_file, 'r') as file:
            print(f"Analyzing sections in {preamble_file} for gradient mode (comment='{comment_char}')")
            for line_num, line in enumerate(file, 1):
                line_stripped = line.strip()

                # Handle scheme block - skip it
                if '!scheme' in line_stripped.lower():
                    in_scheme_block = True
                    continue
                if '!end' in line_stripped.lower():
                    in_scheme_block = False
                    continue
                if in_scheme_block:
                    continue

                # Detect new section marker
                if line_stripped.startswith(comment_char) and "SECTION" in line_stripped.upper():
                    current_section_id = line_stripped
                    section_metadata[current_section_id] = {
                        'dens': 0,  # Default value
                        'has_analytical': False,
                        'content': []
                    }
                    print(f"  Found section marker: '{current_section_id}' at line {line_num}")
                    continue

                # Detect dens= keyword
                if 'dens=' in line_stripped.lower():
                    # Extract value after dens=
                    parts = line_stripped.lower().split('dens=')
                    if len(parts) > 1:
                        # Extract the numeric value (handle dens=2, dens= 2, etc.)
                        value_str = parts[1].split()[0] if parts[1].split() else ''
                        try:
                            dens_value = int(value_str)
                            section_metadata[current_section_id]['dens'] = dens_value
                            section_metadata[current_section_id]['has_analytical'] = (dens_value == 2)
                            print(f"    Section '{current_section_id}': dens={dens_value} (analytical={dens_value == 2}) at line {line_num}")
                        except (ValueError, IndexError):
                            print(f"    Warning: Could not parse dens value at line {line_num}: '{line_stripped}'")

                # Store the content line
                if current_section_id in section_metadata:
                    section_metadata[current_section_id]['content'].append(line)

        # Summary
        print("\n--- Section Analysis Summary ---")
        analytical_sections = []
        numerical_sections = []
        for section_id, metadata in section_metadata.items():
            status = "ANALYTICAL" if metadata['has_analytical'] else "NUMERICAL"
            print(f"  {section_id}: {status} (dens={metadata['dens']}, lines={len(metadata['content'])})")
            if metadata['has_analytical']:
                analytical_sections.append(section_id)
            else:
                numerical_sections.append(section_id)

        print(f"\nTotal sections: {len(section_metadata)}")
        print(f"Analytical sections ({len(analytical_sections)}): {analytical_sections}")
        print(f"Numerical sections ({len(numerical_sections)}): {numerical_sections}")
        print("--------------------------------")

        return section_metadata

    except FileNotFoundError:
        print(f"Warning: Preamble file '{preamble_file}' not found.")
        return {'default': {'dens': 0, 'has_analytical': False, 'content': []}}
    except Exception as e:
        print(f"Error analyzing sections in '{preamble_file}': {e}")
        return {'default': {'dens': 0, 'has_analytical': False, 'content': []}}


def split_preamble_by_gradient_type(preamble_file, section_metadata, output_dir, program='mrcc'):
    """
    Split preamble file into two versions: one for analytical gradients, one for numerical.
    Preserves the scheme block in both files, adjusting coefficient indices as needed.

    Args:
        preamble_file (str): Path to original preamble file
        section_metadata (dict): Section metadata from analyze_sections_for_gradient_mode()
        output_dir (str): Directory where split files will be written
        program (str): Program type ('mrcc', 'molpro' use '!', others use '#')

    Returns:
        tuple: (analytical_preamble_path, numerical_preamble_path, analytical_indices, numerical_indices)
            - analytical_preamble_path: Path to analytical gradients preamble file
            - numerical_preamble_path: Path to numerical gradients preamble file
            - analytical_indices: List of original section indices that are analytical
            - numerical_indices: List of original section indices that are numerical
    """
    import os

    comment_char = '!' if program.lower() in ['mrcc', 'molpro', 'gaussian'] else '#'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    analytical_preamble_path = os.path.join(output_dir, "preamble_analytical.dat")
    numerical_preamble_path = os.path.join(output_dir, "preamble_numerical.dat")

    # Extract scheme/formula information from original file
    original_coefficients = []
    original_formula = None
    in_scheme_block = False
    scheme_header = None
    scheme_footer = None

    try:
        with open(preamble_file, 'r') as f:
            for line in f:
                line_stripped = line.strip()
                if '!scheme' in line_stripped.lower():
                    in_scheme_block = True
                    scheme_header = line  # Preserve original formatting
                    continue
                if '!end' in line_stripped.lower() and in_scheme_block:
                    scheme_footer = line  # Preserve original formatting
                    in_scheme_block = False
                    break
                if in_scheme_block:
                    # Parse coefficients or formula from this line
                    # Remove comment character and whitespace
                    content = line_stripped.lstrip('!').lstrip('#').strip()
                    if content:
                        # Check if this is a formula
                        if '=' in content and any(keyword in content.lower() for keyword in ['sqrt', 'e=']):
                            original_formula = content
                        else:
                            # Parse coefficients (space-separated numbers)
                            parts = content.split()
                            for part in parts:
                                try:
                                    coeff = float(part)
                                    original_coefficients.append(coeff)
                                except ValueError:
                                    continue

        # Classify sections and track indices
        analytical_sections = []
        numerical_sections = []
        analytical_indices = []
        numerical_indices = []

        # Sort sections by their SECTION number, excluding 'default'
        def extract_section_number(section_id):
            """Extract numeric part from !SECTION1, !SECTION2, etc."""
            if 'SECTION' in section_id.upper():
                try:
                    # Extract number after SECTION
                    parts = section_id.upper().split('SECTION')
                    if len(parts) > 1:
                        # Get numeric part, handling cases like "SECTION1" or "SECTION 1"
                        num_str = ''.join(c for c in parts[1] if c.isdigit())
                        return int(num_str) if num_str else 999
                except:
                    return 999
            return 999

        section_order = sorted(
            [k for k in section_metadata.keys() if k != 'default'],
            key=extract_section_number
        )

        for idx, section_id in enumerate(section_order, 1):
            metadata = section_metadata[section_id]
            # Only add sections that have content
            if len(metadata['content']) > 0:
                if metadata['has_analytical']:
                    analytical_sections.append((section_id, metadata))
                    analytical_indices.append(idx)
                else:
                    numerical_sections.append((section_id, metadata))
                    numerical_indices.append(idx)

        # Handle default section (only if it has content beyond the scheme)
        if 'default' in section_metadata:
            default_metadata = section_metadata['default']
            # Only include default if it has meaningful content (more than just whitespace/comments)
            has_content = any(line.strip() and not line.strip().startswith('!')
                            for line in default_metadata['content'])
            if has_content:
                if default_metadata['has_analytical']:
                    analytical_sections.insert(0, ('default', default_metadata))
                    analytical_indices.insert(0, 0)
                else:
                    numerical_sections.insert(0, ('default', default_metadata))
                    numerical_indices.insert(0, 0)

        # Create reduced schemes for each file
        # Extract only coefficients for sections present in each file
        analytical_coefficients = []
        numerical_coefficients = []

        # Filter indices to only include non-default sections (idx > 0)
        analytical_nondefault_indices = [idx for idx in analytical_indices if idx > 0]
        numerical_nondefault_indices = [idx for idx in numerical_indices if idx > 0]

        print(f"\n--- Creating Reduced Schemes ---")
        print(f"Original coefficients: {original_coefficients}")
        print(f"Original formula: {original_formula}")
        print(f"Analytical section indices (non-default): {analytical_nondefault_indices}")
        print(f"Numerical section indices (non-default): {numerical_nondefault_indices}")

        # Extract coefficients for analytical sections
        for idx in analytical_nondefault_indices:
            if idx <= len(original_coefficients):
                analytical_coefficients.append(original_coefficients[idx - 1])

        # Extract coefficients for numerical sections
        for idx in numerical_nondefault_indices:
            if idx <= len(original_coefficients):
                numerical_coefficients.append(original_coefficients[idx - 1])

        print(f"Analytical coefficients: {analytical_coefficients}")
        print(f"Numerical coefficients: {numerical_coefficients}")

        # Write analytical preamble
        print(f"\n--- Writing Analytical Preamble to {analytical_preamble_path} ---")
        with open(analytical_preamble_path, 'w') as f:
            # Write reduced scheme block
            if scheme_header and scheme_footer:
                f.write(scheme_header)
                if original_formula:
                    # For formulas, keep the original formula but user needs to be aware
                    # that only certain sections are present
                    f.write(f"{comment_char}  {original_formula}\n")
                elif analytical_coefficients:
                    # Write only coefficients for analytical sections
                    coeff_str = '  '.join(str(c) for c in analytical_coefficients)
                    f.write(f"{comment_char}  {coeff_str}\n")
                f.write(scheme_footer)

            # Write analytical sections with renumbered section markers
            # Renumber sections sequentially starting from 1 to match reduced scheme
            section_counter = 1
            for section_id, metadata in analytical_sections:
                if section_id != 'default':
                    # Write renumbered section marker
                    f.write(f"\n{comment_char}SECTION{section_counter}\n")
                    section_counter += 1
                for line in metadata['content']:
                    f.write(line)
        print(f"  Wrote {len(analytical_sections)} analytical section(s)")
        print(f"  Analytical indices (original): {analytical_indices}")
        print(f"  Sections renumbered: 1 to {section_counter - 1}")
        print(f"  Reduced scheme: {analytical_coefficients if not original_formula else original_formula}")

        # Write numerical preamble
        print(f"\n--- Writing Numerical Preamble to {numerical_preamble_path} ---")
        with open(numerical_preamble_path, 'w') as f:
            # Write reduced scheme block
            if scheme_header and scheme_footer:
                f.write(scheme_header)
                if original_formula:
                    # For formulas, keep the original formula but user needs to be aware
                    # that only certain sections are present
                    f.write(f"{comment_char}  {original_formula}\n")
                elif numerical_coefficients:
                    # Write only coefficients for numerical sections
                    coeff_str = '  '.join(str(c) for c in numerical_coefficients)
                    f.write(f"{comment_char}  {coeff_str}\n")
                f.write(scheme_footer)

            # Write numerical sections with renumbered section markers
            # Renumber sections sequentially starting from 1 to match reduced scheme
            section_counter = 1
            for section_id, metadata in numerical_sections:
                if section_id != 'default':
                    # Write renumbered section marker
                    f.write(f"\n{comment_char}SECTION{section_counter}\n")
                    section_counter += 1
                for line in metadata['content']:
                    f.write(line)
        print(f"  Wrote {len(numerical_sections)} numerical section(s)")
        print(f"  Numerical indices (original): {numerical_indices}")
        print(f"  Sections renumbered: 1 to {section_counter - 1}")
        print(f"  Reduced scheme: {numerical_coefficients if not original_formula else original_formula}")

        print("--- Preamble Splitting Complete ---\n")

        return analytical_preamble_path, numerical_preamble_path, analytical_indices, numerical_indices

    except Exception as e:
        print(f"Error splitting preamble file '{preamble_file}': {e}")
        raise


def combine_mixed_gradients(analytical_gradients, numerical_gradients, analytical_indices, numerical_indices, operations, formula=None):
    """
    Combine analytical and numerical gradients according to the scheme coefficients and formula.

    Args:
        analytical_gradients (list): List of gradient arrays from analytical calculations
        numerical_gradients (list): List of gradient arrays from numerical calculations
        analytical_indices (list): Original indices of analytical sections
        numerical_indices (list): Original indices of numerical sections
        operations (list): Parsed operations from parse_scheme_and_formula()
        formula (str, optional): Custom formula for gradient combination

    Returns:
        np.ndarray: Combined gradient array (N_atoms x 3)
    """
    import numpy as np

    # Reconstruct full gradient list in original order
    num_sections = len(analytical_indices) + len(numerical_indices)
    all_gradients = [None] * num_sections

    # Place analytical gradients at their original indices
    for i, idx in enumerate(analytical_indices):
        if idx > 0:  # Skip index 0 (default section)
            all_gradients[idx - 1] = analytical_gradients[i]

    # Place numerical gradients at their original indices
    for i, idx in enumerate(numerical_indices):
        if idx > 0:  # Skip index 0 (default section)
            all_gradients[idx - 1] = numerical_gradients[i]

    # Remove None entries (from default section)
    all_gradients = [g for g in all_gradients if g is not None]

    print(f"\n--- Combining Mixed Gradients ---")
    print(f"Analytical indices: {analytical_indices}")
    print(f"Numerical indices: {numerical_indices}")
    print(f"Total gradient components: {len(all_gradients)}")

    # If custom formula is provided, use it
    if formula:
        print(f"Using custom formula: {formula}")
        # Extract coefficients from operations
        coefficients = []
        for op in operations:
            if op[0] == 'coeff':
                coefficients.append(op[1])
            elif op[0] == 'copy':
                coefficients.append(op[2])

        # Use evaluate_custom_formula for gradient combination
        combined_gradient = evaluate_custom_formula(formula, all_gradients, coefficients)
        print(f"Combined gradient using formula")
    else:
        # Use linear combination based on operations
        print("Using linear combination based on scheme operations")

        # Initialize combined gradient
        combined_gradient = np.zeros_like(all_gradients[0], dtype=float)

        # Process operations
        gradient_index = 0
        for op in operations:
            if op[0] == 'coeff':
                # Simple coefficient multiplication
                coeff = op[1]
                if gradient_index < len(all_gradients):
                    combined_gradient += coeff * all_gradients[gradient_index]
                    print(f"  Operation: gradient[{gradient_index}] * {coeff}")
                    gradient_index += 1
            elif op[0] == 'copy':
                # Copy from another gradient with coefficient
                source_index = op[1] - 1  # Convert to 0-indexed
                coeff = op[2]
                if source_index < len(all_gradients):
                    combined_gradient += coeff * all_gradients[source_index]
                    print(f"  Operation: copy gradient[{source_index}] * {coeff}")

    print(f"Combined gradient shape: {combined_gradient.shape}")
    print("--- Gradient Combination Complete ---\n")

    return combined_gradient


def combine_mixed_energies(analytical_energies, numerical_energies, analytical_indices, numerical_indices, operations, formula=None):
    """
    Combine analytical and numerical energies according to the scheme coefficients and formula.

    Args:
        analytical_energies (list): List of energy values from analytical calculations
        numerical_energies (list): List of energy values from numerical calculations
        analytical_indices (list): Original indices of analytical sections
        numerical_indices (list): Original indices of numerical sections
        operations (list): Parsed operations from parse_scheme_and_formula()
        formula (str, optional): Custom formula for energy combination

    Returns:
        float: Combined energy value
    """
    # Reconstruct full energy list in original order
    num_sections = len(analytical_indices) + len(numerical_indices)
    all_energies = [None] * num_sections

    # Place analytical energies at their original indices
    for i, idx in enumerate(analytical_indices):
        if idx > 0:  # Skip index 0 (default section)
            all_energies[idx - 1] = analytical_energies[i]

    # Place numerical energies at their original indices
    for i, idx in enumerate(numerical_indices):
        if idx > 0:  # Skip index 0 (default section)
            all_energies[idx - 1] = numerical_energies[i]

    # Remove None entries (from default section)
    all_energies = [e for e in all_energies if e is not None]

    print(f"\n--- Combining Mixed Energies ---")
    print(f"Analytical energies: {analytical_energies}")
    print(f"Numerical energies: {numerical_energies}")
    print(f"Reconstructed order: {all_energies}")

    # If custom formula is provided, use it
    if formula:
        print(f"Using custom formula: {formula}")
        # Extract coefficients from operations
        coefficients = []
        for op in operations:
            if op[0] == 'coeff':
                coefficients.append(op[1])
            elif op[0] == 'copy':
                coefficients.append(op[2])

        # Use evaluate_custom_formula for energy combination
        combined_energy = evaluate_custom_formula(formula, all_energies, coefficients)
        print(f"Combined energy using formula: {combined_energy}")
    else:
        # Use linear combination based on operations
        print("Using linear combination based on scheme operations")
        combined_energy = 0.0

        # Process operations
        energy_index = 0
        for op in operations:
            if op[0] == 'coeff':
                # Simple coefficient multiplication
                coeff = op[1]
                if energy_index < len(all_energies):
                    combined_energy += coeff * all_energies[energy_index]
                    print(f"  Operation: energy[{energy_index}] * {coeff} = {coeff * all_energies[energy_index]}")
                    energy_index += 1
            elif op[0] == 'copy':
                # Copy from another energy with coefficient
                source_index = op[1] - 1  # Convert to 0-indexed
                coeff = op[2]
                if source_index < len(all_energies):
                    combined_energy += coeff * all_energies[source_index]
                    print(f"  Operation: copy energy[{source_index}] * {coeff} = {coeff * all_energies[source_index]}")

        print(f"Combined energy: {combined_energy}")

    print("--- Energy Combination Complete ---\n")

    return combined_energy


def parse_raw_analytical_results(file_path, atoms):
    """
    Parse the raw_analytical_results.dat file to extract individual section
    energies and gradients before coefficient application.

    Args:
        file_path (str): Path to raw_analytical_results.dat
        atoms (int): Expected number of atoms for gradient validation

    Returns:
        list: List of dicts with 'energy', 'gradient', 'coefficient' for each section.
              Returns empty list if file doesn't exist or parse fails.
    """
    import numpy as np

    results = []

    if not os.path.exists(file_path):
        print(f"WARNING: Raw analytical results file not found: {file_path}")
        return results

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        i = 0
        current_section = None

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Check for header
            if line.startswith("# RAW_ANALYTICAL_RESULTS"):
                i += 1
                continue

            # Check for section count
            if line.startswith("# SECTION_COUNT:"):
                section_count = int(line.split(":")[1].strip())
                print(f"Parsing {section_count} raw analytical sections from {file_path}")
                i += 1
                continue

            # Check for section marker
            if line.startswith("# SECTION"):
                current_section = {
                    'energy': 0.0,
                    'gradient': None,
                    'coefficient': 1.0
                }
                results.append(current_section)
                i += 1
                continue

            # Parse coefficient
            if line.startswith("COEFFICIENT:") and current_section is not None:
                current_section['coefficient'] = float(line.split(":")[1].strip())
                i += 1
                continue

            # Parse energy
            if line.startswith("ENERGY:") and current_section is not None:
                current_section['energy'] = float(line.split(":")[1].strip())
                i += 1
                continue

            # Parse gradient
            if line.startswith("GRADIENT:") and current_section is not None:
                if "NONE" in line:
                    current_section['gradient'] = None
                    i += 1
                    continue
                else:
                    # Read gradient lines
                    gradient = np.zeros((atoms, 3))
                    i += 1
                    for atom_idx in range(atoms):
                        if i < len(lines):
                            grad_line = lines[i].strip()
                            if grad_line and not grad_line.startswith("#"):
                                parts = grad_line.split()
                                if len(parts) >= 3:
                                    gradient[atom_idx] = [float(parts[0]), float(parts[1]), float(parts[2])]
                            i += 1
                    current_section['gradient'] = gradient
                    continue

            i += 1

        print(f"Successfully parsed {len(results)} raw analytical results")
        return results

    except Exception as e:
        print(f"ERROR parsing raw analytical results from {file_path}: {e}")
        return []
