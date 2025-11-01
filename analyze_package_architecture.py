#!/usr/bin/env python3
"""
Analyze arz_model package architecture for violations of:
1. Package Principles (Cohesion & Coupling)
2. Clean Architecture layers
3. Dependency direction
4. Circular dependencies
"""

import os
import ast
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def extract_imports(filepath: Path) -> Tuple[List[str], List[str]]:
    """Extract imports and from-imports from a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))
        
        imports = []
        from_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    from_imports.append(node.module)
        
        return imports, from_imports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return [], []

def analyze_package_structure(root_path: Path) -> Dict:
    """Analyze entire package structure"""
    
    # Collect all Python files
    python_files = {}
    module_dependencies = defaultdict(set)
    
    for py_file in root_path.rglob('*.py'):
        if '__pycache__' in str(py_file) or 'test' in str(py_file).lower():
            continue
        
        rel_path = py_file.relative_to(root_path)
        module_path = str(rel_path).replace(os.sep, '.').replace('.py', '')
        
        # Count lines
        with open(py_file, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        
        # Extract imports
        imports, from_imports = extract_imports(py_file)
        
        # Filter for internal imports only (starting with 'arz_model')
        internal_deps = set()
        for imp in imports + from_imports:
            if imp and imp.startswith('arz_model'):
                # Get top-level module
                parts = imp.split('.')
                if len(parts) >= 2:
                    top_module = parts[1]  # e.g., 'core', 'simulation', etc.
                    internal_deps.add(top_module)
        
        python_files[module_path] = {
            'filepath': str(rel_path),
            'lines': lines,
            'imports': imports + from_imports,
            'internal_deps': list(internal_deps)
        }
        
        # Track module-to-module dependencies
        path_parts = module_path.split('.')
        if len(path_parts) >= 1 and path_parts[0]:
            current_module = path_parts[0]
        else:
            current_module = 'root'
        
        for dep in internal_deps:
            if dep != current_module:
                module_dependencies[current_module].add(dep)
    
    return {
        'files': python_files,
        'module_deps': {k: list(v) for k, v in module_dependencies.items()},
        'total_files': len(python_files),
        'total_lines': sum(f['lines'] for f in python_files.values())
    }

def detect_circular_dependencies(module_deps: Dict[str, List[str]]) -> List[List[str]]:
    """Detect circular dependencies using DFS"""
    
    def dfs(node, path, visited, rec_stack):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in module_deps.get(node, []):
            if neighbor not in visited:
                cycle = dfs(neighbor, path.copy(), visited, rec_stack)
                if cycle:
                    return cycle
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
        
        rec_stack.remove(node)
        return None
    
    cycles = []
    visited = set()
    
    for node in module_deps:
        if node not in visited:
            cycle = dfs(node, [], visited, set())
            if cycle and cycle not in cycles:
                cycles.append(cycle)
    
    return cycles

def calculate_package_metrics(module_deps: Dict[str, List[str]]) -> Dict:
    """Calculate coupling metrics"""
    
    # Afferent coupling (Ca): number of modules that depend on this module
    # Efferent coupling (Ce): number of modules this module depends on
    
    afferent = defaultdict(int)
    efferent = defaultdict(int)
    
    all_modules = set(module_deps.keys())
    for deps in module_deps.values():
        all_modules.update(deps)
    
    for module in all_modules:
        # Count afferent (who depends on me?)
        for m, deps in module_deps.items():
            if module in deps:
                afferent[module] += 1
        
        # Count efferent (who do I depend on?)
        efferent[module] = len(module_deps.get(module, []))
    
    # Instability = Ce / (Ce + Ca)  [0 = stable, 1 = instable]
    instability = {}
    for module in all_modules:
        ca = afferent[module]
        ce = efferent[module]
        total = ca + ce
        instability[module] = ce / total if total > 0 else 0
    
    return {
        'afferent': dict(afferent),
        'efferent': dict(efferent),
        'instability': instability
    }

def main():
    root = Path('arz_model')
    
    print("=" * 80)
    print("ARZ_MODEL PACKAGE ARCHITECTURE ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze structure
    analysis = analyze_package_structure(root)
    
    print(f"📦 PACKAGE SUMMARY")
    print(f"   Total Python files: {analysis['total_files']}")
    print(f"   Total lines of code: {analysis['total_lines']}")
    print()
    
    # Module dependencies
    print(f"🔗 MODULE DEPENDENCIES")
    for module, deps in sorted(analysis['module_deps'].items()):
        if deps:
            print(f"   {module} → {', '.join(deps)}")
    print()
    
    # Detect circular dependencies
    cycles = detect_circular_dependencies(analysis['module_deps'])
    print(f"⚠️  CIRCULAR DEPENDENCIES DETECTED: {len(cycles)}")
    for i, cycle in enumerate(cycles, 1):
        print(f"   Cycle {i}: {' → '.join(cycle)}")
    print()
    
    # Package metrics
    metrics = calculate_package_metrics(analysis['module_deps'])
    
    print(f"📊 PACKAGE COUPLING METRICS")
    print(f"   {'Module':<20} {'Afferent (Ca)':<15} {'Efferent (Ce)':<15} {'Instability':<12}")
    print(f"   {'-'*20} {'-'*15} {'-'*15} {'-'*12}")
    
    all_modules = sorted(set(list(metrics['afferent'].keys()) + list(metrics['efferent'].keys())))
    for module in all_modules:
        ca = metrics['afferent'].get(module, 0)
        ce = metrics['efferent'].get(module, 0)
        inst = metrics['instability'].get(module, 0)
        status = "⚠️ UNSTABLE" if inst > 0.5 else "✅ STABLE"
        print(f"   {module:<20} {ca:<15} {ce:<15} {inst:>6.2f} {status}")
    print()
    
    # Largest files
    print(f"📏 LARGEST FILES (God Object smell)")
    sorted_files = sorted(analysis['files'].items(), key=lambda x: x[1]['lines'], reverse=True)
    for i, (module, data) in enumerate(sorted_files[:10], 1):
        status = "⚠️ TOO LARGE" if data['lines'] > 300 else "✅ OK"
        print(f"   {i}. {data['filepath']:<50} {data['lines']:>5} lines {status}")
    print()
    
    # Save full analysis
    with open('package_architecture_analysis.json', 'w') as f:
        json.dump({
            'summary': {
                'total_files': analysis['total_files'],
                'total_lines': analysis['total_lines']
            },
            'module_dependencies': analysis['module_deps'],
            'circular_dependencies': cycles,
            'coupling_metrics': metrics,
            'files': analysis['files']
        }, f, indent=2)
    
    print("✅ Full analysis saved to: package_architecture_analysis.json")
    print()

if __name__ == '__main__':
    main()
