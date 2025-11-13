"""
Enhanced Architecture Analysis Script v3.1
==========================================

Fixes from v3:
1. Fixed pathlib.relative_to bug → Use py_file.as_posix()
2. Move file_relative before try/except to avoid NameError
3. Full module label for parent_path matching
4. Merged all_decorated set for precise FP detection

Author: Enhanced by Grok (xAI)
Date: 2025-11-13
"""

import json
from pathlib import Path
from collections import defaultdict
import sys
import re
import ast

# Redirect output to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open output file
output_file = open('arz_model/architecture_analysis_v3.txt', 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, output_file)

print("=" * 80)
print("🧠 ENHANCED ARCHITECTURE ANALYSIS V3.1")
print("=" * 80)
print("\nFixes from v3:")
print("  ✓ Fixed pathlib.relative_to → py_file.as_posix()")
print("  ✓ file_relative before try/except (no NameError)")
print("  ✓ Full module labels for decorator matching")
print("  ✓ Merged all_decorated set for FPs")
print("=" * 80)

# Load the dependency graph directly from depviz.json
print("\n📂 Loading dependency graph...")
with open('depviz.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get all nodes for arz_model (modules, functions, classes)
arz_nodes = [
    n for n in data['nodes'] 
    if ('arz_model/' in n.get('id', '') or 'arz_model/' in n.get('fsPath', ''))
    and 'arz_model_gpu/' not in n.get('id', '')
    and 'arz_model_gpu/' not in n.get('fsPath', '')
]

modules = [n for n in arz_nodes if n.get('kind') == 'module']
functions = [n for n in arz_nodes if n.get('kind') == 'func']
classes = [n for n in arz_nodes if n.get('kind') == 'class']

print(f"\n📦 LOADED:")
print(f"  - {len(modules)} modules")
print(f"  - {len(functions)} functions")
print(f"  - {len(classes)} classes")

# ==============================================================================
# ENHANCED DECORATOR DETECTION (v3.1: Fixed paths)
# ==============================================================================

print("\n" + "=" * 80)
print("🔍 SCANNING FOR DECORATORS AND SPECIAL PATTERNS (v3.1 FIXED)")
print("=" * 80)

# Scan actual Python files for decorators
decorator_functions = set()  # (file, func) tuples
pytest_tests = set()
pytest_fixtures = set()
pydantic_validators = set()
property_methods = set()
classmethod_functions = set()
staticmethod_functions = set()
main_entry_points = set()

# Scan all Python files in arz_model
arz_model_path = Path('arz_model')
python_files = list(arz_model_path.rglob('*.py'))

print(f"\n🔬 Scanning {len(python_files)} Python files for decorators...")

# DEBUG: Track what we find
debug_decorators = []
debug_pytest_tests = []
debug_main_calls = []
parse_errors = []

for py_file in python_files:
    file_relative = py_file.as_posix()  # v3.1: Simple & robust
    try:
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Detect functions with decorators
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                
                # Check for decorators
                for decorator in node.decorator_list:
                    decorator_name = ''
                    if isinstance(decorator, ast.Name):
                        decorator_name = decorator.id
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorator_name = decorator.func.id
                        elif isinstance(decorator.func, ast.Attribute):
                            # v3: Get full qualified name
                            if isinstance(decorator.func.value, ast.Name):
                                decorator_name = decorator.func.value.id + '.' + decorator.func.attr
                            else:
                                decorator_name = decorator.func.attr
                        # Handle nested (fallback)
                        if not decorator_name and isinstance(decorator.func, ast.Attribute):
                            decorator_name = decorator.func.attr
                    
                    if decorator_name:
                        debug_decorators.append((func_name, decorator_name, file_relative))
                        
                        # Record decorator type
                        full_dec_name = decorator_name.lower()
                        if 'pytest.fixture' in full_dec_name or 'fixture' in full_dec_name:
                            pytest_fixtures.add((file_relative, func_name))
                        elif 'field_validator' in full_dec_name or 'validator' in full_dec_name:
                            pydantic_validators.add((file_relative, func_name))
                        elif 'property' in full_dec_name:
                            property_methods.add((file_relative, func_name))
                        elif 'classmethod' in full_dec_name:
                            classmethod_functions.add((file_relative, func_name))
                        elif 'staticmethod' in full_dec_name:
                            staticmethod_functions.add((file_relative, func_name))
                        
                        decorator_functions.add((file_relative, func_name))
                
                # Check for pytest test functions
                if func_name.startswith('test_'):
                    debug_pytest_tests.append((func_name, file_relative))
                    pytest_tests.add((file_relative, func_name))
            
            # Detect if __name__ == '__main__' blocks (enhanced)
            if isinstance(node, ast.If):
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and
                    any(isinstance(op, ast.Eq) for op in node.test.ops) and
                    len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant) and
                    node.test.comparators[0].value == '__main__'):
                    
                    # Find function calls in this block
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            called_name = ''
                            if isinstance(child.func, ast.Name):
                                called_name = child.func.id
                            elif isinstance(child.func, ast.Attribute):
                                called_name = child.func.attr
                            if called_name:
                                debug_main_calls.append((called_name, file_relative))
                                main_entry_points.add((file_relative, called_name))
    
    except SyntaxError as e:
        parse_errors.append((file_relative, str(e)))
    except Exception as e:
        parse_errors.append((file_relative, str(e)))

# DEBUG: Print what we found
print(f"\n🔍 DEBUG - Decorators found: {len(debug_decorators)}")
for func_name, decorator_name, file_path in debug_decorators[:10]:
    print(f"  ✓ {func_name} @{decorator_name} in {file_path}")

if parse_errors:
    print(f"\n⚠️  {len(parse_errors)} files failed to parse:")
    for file_path, err in parse_errors[:5]:
        print(f"  - {file_path}: {err}")

print(f"\n🔍 DEBUG - Pytest tests found: {len(debug_pytest_tests)}")
for func_name, file_path in debug_pytest_tests[:10]:
    print(f"  ✓ {func_name} in {file_path}")

print(f"\n🔍 DEBUG - Main calls found: {len(debug_main_calls)}")
for func_name, file_path in debug_main_calls[:10]:
    print(f"  ✓ {func_name} called from main in {file_path}")

print(f"\n📊 DECORATOR DETECTION RESULTS:")
print(f"  - {len(decorator_functions)} functions with decorators")
print(f"  - {len(pytest_tests)} pytest test functions (test_*)")
print(f"  - {len(pytest_fixtures)} pytest fixtures (@pytest.fixture)")
print(f"  - {len(pydantic_validators)} Pydantic validators (@field_validator)")
print(f"  - {len(property_methods)} property methods (@property)")
print(f"  - {len(classmethod_functions)} class methods (@classmethod)")
print(f"  - {len(staticmethod_functions)} static methods (@staticmethod)")
print(f"  - {len(main_entry_points)} main entry point calls (if __name__ == '__main__')")

# ==============================================================================
# BUILD ENHANCED CALL GRAPH (v3: Relaxed for /execution/)
# ==============================================================================

print("\n" + "=" * 80)
print("🔗 BUILDING ENHANCED CALL GRAPH")
print("=" * 80)

# Filter edges for arz_model (relaxed - include if either in arz_model)
edges = data.get('edges', [])
arz_edges = []

for e in edges:
    from_id = e.get('from', '')
    to_id = e.get('to', '')
    
    # Include if either endpoint is in arz_model (cross-module OK)
    if from_id and to_id:
        if ('arz_model/' in from_id or 'arz_model/' in to_id) and 'arz_model_gpu/' not in from_id and 'arz_model_gpu/' not in to_id:
            arz_edges.append(e)

print(f"Total dependencies: {len(arz_edges)}")

# Build call graph
calls_to = defaultdict(list)
called_by = defaultdict(list)

for edge in arz_edges:
    source = edge.get('from')
    target = edge.get('to')
    if source and target:
        calls_to[source].append(target)
        called_by[target].append(source)

print(f"  - {len(calls_to)} functions that call others")
print(f"  - {len(called_by)} functions that are called")

# Find all arz_model function/class IDs
arz_func_ids = {n['id'] for n in functions}
arz_class_ids = {n['id'] for n in classes}

# ==============================================================================
# ENHANCED DEAD CODE DETECTION (v3.1: Full path matching)
# ==============================================================================

print("\n" + "=" * 80)
print("⚠️  ENHANCED DEAD CODE DETECTION")
print("=" * 80)

# Identify dead code
all_callables = arz_func_ids | arz_class_ids
called_at_least_once = {target for target in called_by.keys() if target in all_callables}
never_called = all_callables - called_at_least_once

# Merged set for all special functions (tuples: (file, func))
all_decorated = (decorator_functions | pytest_tests | pytest_fixtures | 
                 pydantic_validators | property_methods | classmethod_functions | 
                 staticmethod_functions | main_entry_points)

def is_suspected_false_positive(node_id, label, parent_label):
    """Detect if a function is likely a false positive"""
    
    # Check if decorated/special (full path match)
    if (parent_label, label) in all_decorated:
        return True, "Decorated/special function (pytest/field_validator/property/etc.)"
    
    # Pytest by name (fallback)
    if label.startswith('test_'):
        return True, "Pytest test function (test_*)"
    
    # Public API patterns
    public_api_patterns = ['run', 'get_', 'set_', 'execute', 'initialize', 'create_', 'build_']
    if any(pattern in label.lower() for pattern in public_api_patterns):
        return True, "Public API method (run/get/set/execute/etc.)"
    
    # GPU kernel
    if label.endswith('_kernel'):
        return True, "GPU kernel (called indirectly via classes)"
    
    # Main-like
    if 'main' in label.lower():
        return True, "Main entry point candidate"
    
    return False, None

def is_regular_function(node_id):
    """Check if a node is a regular function (not dunder)"""
    matching = [n for n in arz_nodes if n['id'] == node_id]
    if not matching:
        return False
    label = matching[0].get('label', '')
    
    # Exclude dunder methods and __init__
    if (label.startswith('__') and label.endswith('__')) or label == '__init__':
        return False
    
    return True

# Categorize dead functions
confirmed_dead = []
suspected_false_positives = []

for node_id in never_called:
    if node_id in arz_func_ids and is_regular_function(node_id):
        matching = [n for n in functions if n['id'] == node_id]
        if matching:
            func = matching[0]
            label = func.get('label', 'unknown')
            parent = func.get('parent', 'unknown')
            
            # v3.1: Get FULL parent module label
            parent_label = 'unknown'
            for m in modules:
                if m['id'] == parent:
                    parent_label = m['label']  # Full 'arz_model/some/file.py'
                    break
            
            # Check if suspected false positive
            is_fp, reason = is_suspected_false_positive(node_id, label, parent_label)
            
            if is_fp:
                suspected_false_positives.append({
                    'label': label,
                    'module': parent_label,
                    'reason': reason
                })
            else:
                confirmed_dead.append({
                    'label': label,
                    'module': parent_label
                })

print(f"\n📊 RESULTS:")
print(f"  - {len(confirmed_dead)} CONFIRMED DEAD functions")
print(f"  - {len(suspected_false_positives)} SUSPECTED FALSE POSITIVES")
print(f"  - Total analyzed: {len(never_called)}")

# ==============================================================================
# REPORT: SUSPECTED FALSE POSITIVES
# ==============================================================================

print("\n" + "=" * 80)
print("🚨 SUSPECTED FALSE POSITIVES (DO NOT DELETE)")
print("=" * 80)
print("\nThese functions appear 'dead' but are likely called via:")
print("  - Decorators (@pytest.fixture, @field_validator, @property)")
print("  - Indirect class calls (e.g., SSP_RK3_GPU().integrate())")
print("  - Main entry points (if __name__ == '__main__')")
print("  - Public API conventions (run, get_*, set_*)")
print("=" * 80)

# Group by reason
by_reason = defaultdict(list)
for fp in suspected_false_positives:
    by_reason[fp['reason']].append(fp)

for reason in sorted(by_reason.keys()):
    fps = by_reason[reason]
    print(f"\n🔍 {reason} ({len(fps)} functions)")
    print("-" * 80)
    
    # Group by module
    by_module = defaultdict(list)
    for fp in fps:
        by_module[fp['module']].append(fp['label'])
    
    for module_label in sorted(by_module.keys()):
        funcs = by_module[module_label]
        print(f"\n   📄 {module_label}")
        for func in sorted(funcs):
            print(f"      ✓ {func}")

# ==============================================================================
# REPORT: CONFIRMED DEAD FUNCTIONS
# ==============================================================================

print("\n" + "=" * 80)
print("❌ CONFIRMED DEAD FUNCTIONS (SAFE TO DELETE)")
print("=" * 80)
print("\nThese functions are NOT called anywhere and have no decorators.")
print("They are likely legacy code that can be safely removed.")
print("=" * 80)

# Group by module
dead_by_module = defaultdict(list)
for dead in confirmed_dead:
    dead_by_module[dead['module']].append(dead['label'])

print(f"\n📊 Total: {len(confirmed_dead)} functions across {len(dead_by_module)} modules\n")

for module_label in sorted(dead_by_module.keys()):
    funcs = dead_by_module[module_label]
    print(f"\n   📄 {module_label} ({len(funcs)} dead functions)")
    for func in sorted(funcs):
        print(f"      ❌ {func}")

# ==============================================================================
# ADDITIONAL INSIGHTS
# ==============================================================================

print("\n" + "=" * 80)
print("🔍 ADDITIONAL INSIGHTS")
print("=" * 80)

# Find entry points
print("\n1. ENTRY POINTS (functions likely called externally):")
print("-" * 80)

entry_point_candidates = []
for func_id in arz_func_ids:
    matching = [n for n in functions if n['id'] == func_id]
    if matching:
        label = matching[0].get('label', '')
        parent = matching[0].get('parent')
        parent_label = 'unknown'
        for m in modules:
            if m['id'] == parent:
                parent_label = m['label']
                break
        
        if any(pattern in label.lower() for pattern in ['main', 'run', 'execute', 'start']):
            caller_count = len(called_by.get(func_id, []))
            entry_point_candidates.append((label, parent_label, caller_count))

entry_point_candidates.sort(key=lambda x: x[2], reverse=True)

for label, module, caller_count in entry_point_candidates[:20]:
    print(f"  {label:40} in {module:40} - called by {caller_count} functions")

# Find hub functions
print("\n\n2. HUB FUNCTIONS (called by many others):")
print("-" * 80)

hub_functions = []
for func_id in called_by.keys():
    if func_id in arz_func_ids:
        matching = [n for n in functions if n['id'] == func_id]
        if matching:
            label = matching[0].get('label', '')
            parent = matching[0].get('parent')
            parent_label = 'unknown'
            for m in modules:
                if m['id'] == parent:
                    parent_label = m['label']
                    break
            
            caller_count = len(called_by[func_id])
            hub_functions.append((label, parent_label, caller_count))

hub_functions.sort(key=lambda x: x[2], reverse=True)

for label, module, caller_count in hub_functions[:30]:
    print(f"  {label:40} ({module:40}) - called by {caller_count} functions")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("📋 SUMMARY")
print("=" * 80)

print(f"""
Architecture Analysis Complete! (v3.1)

Total Functions: {len(functions)}
  ├─ Called at least once: {len(called_at_least_once)}
  ├─ Never called (total): {len(never_called)}
  │   ├─ Confirmed dead: {len(confirmed_dead)}
  │   └─ Suspected false positives: {len(suspected_false_positives)}
  └─ With decorators: {len(decorator_functions)}

Special Functions Detected:
  ├─ Pytest tests: {len(pytest_tests)}
  ├─ Pytest fixtures: {len(pytest_fixtures)}
  ├─ Pydantic validators: {len(pydantic_validators)}
  ├─ Property methods: {len(property_methods)}
  ├─ Class methods: {len(classmethod_functions)}
  ├─ Static methods: {len(staticmethod_functions)}
  └─ Main entry points: {len(main_entry_points)}

Recommendations:
  ✓ DO NOT delete functions in "SUSPECTED FALSE POSITIVES"
  ✓ SAFE to delete functions in "CONFIRMED DEAD FUNCTIONS"
  ✓ Review entry points and hub functions for refactoring opportunities
""")

print("=" * 80)
print("✅ Analysis saved to: arz_model/architecture_analysis_v3.txt")
print("=" * 80)

# Close output file
sys.stdout = original_stdout
output_file.close()

print(f"\n✅ Enhanced analysis complete! (v3.1)")
print(f"📄 Report saved to: arz_model/architecture_analysis_v3.txt")
print(f"\n🔍 Key improvements:")
print(f"  ✓ Fixed path errors - no more crashes")
print(f"  ✓ Detected {len(debug_decorators)} decorators")
print(f"  ✓ {len(suspected_false_positives)} false positives filtered")
print(f"  ✓ {len(confirmed_dead)} truly dead functions")
print(f"  ✓ Scanned {len(python_files)} Python files")