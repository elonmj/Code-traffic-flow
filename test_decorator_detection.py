"""Test decorator detection"""
import ast
from pathlib import Path

test_file = Path('arz_model/tests/test_gpu_memory_pool.py')

with open(test_file, 'r', encoding='utf-8') as f:
    content = f.read()

tree = ast.parse(content)

decorators_found = []

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        func_name = node.name
        
        for decorator in node.decorator_list:
            decorator_name = ''
            if isinstance(decorator, ast.Name):
                decorator_name = decorator.id
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorator_name = decorator.func.id
                elif isinstance(decorator.func, ast.Attribute):
                    decorator_name = decorator.func.attr
            
            if decorator_name:
                decorators_found.append((func_name, decorator_name))
                print(f"✓ Found: {func_name} with @{decorator_name}")

print(f"\n📊 Total: {len(decorators_found)} decorators found")
