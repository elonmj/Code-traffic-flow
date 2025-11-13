#!/usr/bin/env python3
"""
Architecture Analysis Tool for arz_model refactoring

This script processes depviz.json to extract clean architectural information,
removing visual layout data and focusing on code dependencies and structure.

Usage:
    python analyze_architecture.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any


class ArchitectureAnalyzer:
    """Analyzes code architecture from dependency graph"""
    
    def __init__(self, depviz_path: str):
        """Load and parse the dependency visualization JSON"""
        with open(depviz_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        self.nodes = self.raw_data.get('nodes', [])
        self.edges = self.raw_data.get('edges', [])
        
        # Build clean architecture data
        self.clean_nodes = self._clean_nodes()
        self.clean_edges = self._clean_edges()
        self.dependency_graph = self._build_dependency_graph()
    
    def _clean_nodes(self) -> List[Dict[str, Any]]:
        """Remove visual layout data and keep only architectural info"""
        clean = []
        for node in self.nodes:
            # Keep only relevant fields (no x, y, dx, dy, source, etc.)
            clean_node = {
                'id': node.get('id'),
                'kind': node.get('kind'),
                'label': node.get('label'),
            }
            
            # Add optional fields if they exist
            if 'fsPath' in node:
                clean_node['fsPath'] = node['fsPath']
            if 'parent' in node:
                clean_node['parent'] = node['parent']
            
            clean.append(clean_node)
        
        return clean
    
    def _clean_edges(self) -> List[Dict[str, Any]]:
        """Keep only dependency relationships"""
        clean = []
        for edge in self.edges:
            clean_edge = {
                'source': edge.get('source'),
                'target': edge.get('target'),
            }
            
            # Add edge type if available
            if 'type' in edge:
                clean_edge['type'] = edge['type']
            if 'label' in edge:
                clean_edge['label'] = edge['label']
            
            clean.append(clean_edge)
        
        return clean
    
    def _build_dependency_graph(self) -> Dict[str, Dict[str, Any]]:
        """Build a dependency graph for easier querying"""
        graph = defaultdict(lambda: {'dependencies': [], 'dependents': []})
        
        for edge in self.clean_edges:
            source = edge['source']
            target = edge['target']
            
            # source depends on target
            graph[source]['dependencies'].append(target)
            # target is depended on by source
            graph[target]['dependents'].append(source)
        
        # Add node info
        for node in self.clean_nodes:
            node_id = node['id']
            if node_id in graph:
                graph[node_id]['info'] = node
            else:
                graph[node_id] = {
                    'info': node,
                    'dependencies': [],
                    'dependents': []
                }
        
        return dict(graph)
    
    def filter_by_module(self, module_pattern: str) -> Dict[str, Any]:
        """Filter nodes and edges for a specific module (e.g., 'arz_model')"""
        filtered_nodes = [
            node for node in self.clean_nodes
            if module_pattern in (node.get('label') or '') or module_pattern in (node.get('id') or '')
        ]
        
        filtered_node_ids = {node['id'] for node in filtered_nodes}
        
        filtered_edges = [
            edge for edge in self.clean_edges
            if edge['source'] in filtered_node_ids or edge['target'] in filtered_node_ids
        ]
        
        return {
            'nodes': filtered_nodes,
            'edges': filtered_edges,
            'count': {
                'nodes': len(filtered_nodes),
                'edges': len(filtered_edges)
            }
        }
    
    def get_module_structure(self, module_pattern: str) -> Dict[str, List[str]]:
        """Get hierarchical structure of a module"""
        filtered = self.filter_by_module(module_pattern)
        
        structure = defaultdict(list)
        for node in filtered['nodes']:
            label = node.get('label', '')
            if '/' in label:
                parts = label.split('/')
                # Group by first level directory
                if len(parts) > 1:
                    package = parts[0]
                    structure[package].append(label)
            else:
                structure['root'].append(label)
        
        return dict(structure)
    
    def find_circular_dependencies(self, module_pattern: str = None) -> List[List[str]]:
        """Detect circular dependencies in the architecture"""
        # Simple DFS-based cycle detection
        graph = self.dependency_graph
        
        if module_pattern:
            filtered = self.filter_by_module(module_pattern)
            node_ids = {node['id'] for node in filtered['nodes']}
            graph = {k: v for k, v in graph.items() if k in node_ids}
        
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, {}).get('dependencies', []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    def get_dependency_chain(self, node_id: str, depth: int = 3) -> Dict[str, Any]:
        """Get dependency chain for a specific node"""
        result = {
            'node': node_id,
            'dependencies': [],
            'dependents': []
        }
        
        def get_deps(node, current_depth, visited=None):
            if visited is None:
                visited = set()
            
            if current_depth == 0 or node in visited:
                return []
            
            visited.add(node)
            deps = []
            
            for dep in self.dependency_graph.get(node, {}).get('dependencies', []):
                dep_info = self.dependency_graph.get(dep, {}).get('info', {})
                deps.append({
                    'id': dep,
                    'name': dep_info.get('name', dep),
                    'children': get_deps(dep, current_depth - 1, visited.copy())
                })
            
            return deps
        
        result['dependencies'] = get_deps(node_id, depth)
        
        # Get dependents
        def get_dependents(node, current_depth, visited=None):
            if visited is None:
                visited = set()
            
            if current_depth == 0 or node in visited:
                return []
            
            visited.add(node)
            deps = []
            
            for dep in self.dependency_graph.get(node, {}).get('dependents', []):
                dep_info = self.dependency_graph.get(dep, {}).get('info', {})
                deps.append({
                    'id': dep,
                    'name': dep_info.get('name', dep),
                    'children': get_dependents(dep, current_depth - 1, visited.copy())
                })
            
            return deps
        
        result['dependents'] = get_dependents(node_id, depth)
        
        return result
    
    def save_clean_json(self, output_path: str, module_pattern: str = None):
        """Save cleaned architecture to JSON file"""
        if module_pattern:
            data = self.filter_by_module(module_pattern)
        else:
            data = {
                'nodes': self.clean_nodes,
                'edges': self.clean_edges
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved clean architecture to {output_path}")
        print(f"   Nodes: {len(data['nodes'])}")
        print(f"   Edges: {len(data['edges'])}")
    
    def print_summary(self, module_pattern: str = None):
        """Print architectural summary"""
        if module_pattern:
            filtered = self.filter_by_module(module_pattern)
            nodes = filtered['nodes']
            edges = filtered['edges']
            title = f"Architecture Summary for '{module_pattern}'"
        else:
            nodes = self.clean_nodes
            edges = self.clean_edges
            title = "Overall Architecture Summary"
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Total Nodes: {len(nodes)}")
        print(f"Total Edges: {len(edges)}")
        
        # Group by type
        type_counts = defaultdict(int)
        for node in nodes:
            node_type = node.get('kind', 'unknown')
            type_counts[node_type] += 1
        
        print(f"\nNode Types:")
        for node_type, count in sorted(type_counts.items()):
            print(f"  {node_type}: {count}")
        
        # Module structure
        if module_pattern:
            print(f"\nModule Structure:")
            structure = self.get_module_structure(module_pattern)
            for package, files in sorted(structure.items()):
                print(f"\n  📁 {package}/ ({len(files)} files)")
                for file in sorted(files)[:5]:  # Show first 5
                    print(f"     - {file}")
                if len(files) > 5:
                    print(f"     ... and {len(files) - 5} more")


def main():
    """Main entry point"""
    print("🔍 Architecture Analysis Tool")
    print("="*60)
    
    # Load the dependency graph
    depviz_path = Path(__file__).parent / 'depviz.json'
    
    if not depviz_path.exists():
        print(f"❌ Error: {depviz_path} not found")
        return
    
    print(f"📂 Loading: {depviz_path}")
    analyzer = ArchitectureAnalyzer(str(depviz_path))
    
    # Print overall summary
    analyzer.print_summary()
    
    # Focus on arz_model
    print("\n" + "="*60)
    analyzer.print_summary(module_pattern='arz_model')
    
    # Save cleaned versions
    output_dir = Path(__file__).parent / '_arxiv'
    output_dir.mkdir(exist_ok=True)
    
    # Save full clean version
    analyzer.save_clean_json(
        str(output_dir / 'architecture_clean.json')
    )
    
    # Save arz_model only
    analyzer.save_clean_json(
        str(output_dir / 'architecture_arz_model.json'),
        module_pattern='arz_model'
    )
    
    # Save arz_model_gpu only
    analyzer.save_clean_json(
        str(output_dir / 'architecture_arz_model_gpu.json'),
        module_pattern='arz_model_gpu'
    )
    
    # Check for circular dependencies
    print("\n" + "="*60)
    print("🔄 Checking for circular dependencies in arz_model...")
    cycles = analyzer.find_circular_dependencies('arz_model')
    
    if cycles:
        print(f"⚠️  Found {len(cycles)} circular dependencies:")
        for i, cycle in enumerate(cycles[:3], 1):  # Show first 3
            print(f"\n  Cycle {i}:")
            for node in cycle:
                node_info = analyzer.dependency_graph.get(node, {}).get('info', {})
                print(f"    → {node_info.get('name', node)}")
    else:
        print("✅ No circular dependencies found!")
    
    print("\n" + "="*60)
    print("✨ Analysis complete!")
    print(f"\n📊 Clean architecture files saved to {output_dir}/")
    print("\nNext steps for refactoring:")
    print("  1. Review architecture_arz_model.json")
    print("  2. Identify modules to refactor for GPU")
    print("  3. Check dependency chains before moving files")


if __name__ == '__main__':
    main()
