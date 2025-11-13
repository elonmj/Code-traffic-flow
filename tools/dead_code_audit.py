#!/usr/bin/env python3
"""Dead Code Audit Helper V4.

Améliorations:
- AST pour détection précise (définition, décorateurs, appels vs mentions).
- Intégration depviz.json pour cross-check graphe.
- Stats par module (% dead, priorisation).
- Outputs : JSON enrichi, MD avec tables, CSV pour Excel.
"""
from __future__ import annotations

import json
import csv
import ast
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_FILE = REPO_ROOT / "arz_model" / "architecture_analysis_v3.txt"  # Adapté à v3
DEPVIZ_FILE = REPO_ROOT / "depviz.json"
OUTPUT_BASE = REPO_ROOT / "arz_model" / ".copilot-tracking" / "dead-code-audit-20251113-v4"

# Heuristic decorators (étendu)
DECORATOR_WHITELIST = {
    "pytest.fixture", "pytest.mark", "field_validator", "model_validator",
    "property", "staticmethod", "classmethod", "@pytest", "@validator"
}

@dataclass
class DeadFunctionEntry:
    file_path: str
    function_name: str
    decorator: Optional[str] = None
    definition_line: Optional[int] = None
    usage_count: int = 0  # Mentions totales
    call_count: int = 0   # Appels réels
    classification: str = "UNCLASSIFIED"
    notes: List[str] = None
    module_stats: Optional[Dict[str, int]] = None  # % dead par module

    def to_dict(self) -> dict:
        data = asdict(self)
        data["notes"] = self.notes or []
        data["module_stats"] = self.module_stats or {}
        return data

def parse_analysis_file() -> List[DeadFunctionEntry]:
    """Parse robuste de architecture_analysis_v3.txt."""
    entries: List[DeadFunctionEntry] = []
    current_file: Optional[str] = None
    with ANALYSIS_FILE.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            # Regex amélioré pour v3 format
            file_match = re.match(r"^\s*📄\s+(?P<path>arz_model/[^\(]+\.py)", line)
            if file_match:
                current_file = file_match.group("path").strip()
                continue
            if current_file is None:
                continue
            # Match pour ❌ ou espaces
            func_match = re.match(r"^\s*(?:❌|\s{6,})\s+(?P<name>[\w_]+)", line)
            if func_match:
                entries.append(DeadFunctionEntry(
                    file_path=current_file,
                    function_name=func_match.group("name")
                ))
    return entries

def load_dep_graph() -> Dict:
    """Charge depviz.json pour cross-check."""
    if not DEPVIZ_FILE.exists():
        return {"nodes": [], "edges": []}
    with DEPVIZ_FILE.open(encoding="utf-8") as f:
        return json.load(f)

def find_definition_ast(entry: DeadFunctionEntry) -> None:
    """Détection AST pour définition et décorateurs."""
    file_path = REPO_ROOT / entry.file_path
    if not file_path.exists():
        entry.notes = ["Fichier manquant"]
        entry.classification = "MISSING_FILE"
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry.function_name:
                entry.definition_line = node.lineno
                if node.decorator_list:
                    entry.decorator = ast.unparse(node.decorator_list[0])
                return
    except SyntaxError:
        pass

    entry.notes = ["Définition non trouvée via AST"]
    entry.classification = "DEFINITION_NOT_FOUND"

def search_usages_ast(entry: DeadFunctionEntry) -> None:
    """Comptage AST : appels réels vs mentions (plus juste)."""
    if entry.classification in {"MISSING_FILE", "DEFINITION_NOT_FOUND"}:
        return

    call_locations = []
    mention_files = set()

    for py_file in REPO_ROOT.rglob("*.py"):
        if py_file == REPO_ROOT / entry.file_path:
            continue  # Skip self

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                # Appels directs
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry.function_name:
                    call_locations.append(py_file.name)
                # Méthodes/attributs
                elif isinstance(node, ast.Attribute) and node.attr == entry.function_name:
                    call_locations.append(py_file.name)
                # Mentions (imports, vars) via regex fallback
                text = py_file.read_text()
                if re.search(rf"\b{re.escape(entry.function_name)}\b", text):
                    mention_files.add(py_file.name)
        except SyntaxError:
            continue

    entry.call_count = len(set(call_locations))
    entry.usage_count = len(mention_files | set(call_locations))
    if call_locations:
        entry.notes = [f"{len(call_locations)} appels réels détectés"]

def classify_entry_v4(entry: DeadFunctionEntry, dep_graph: Dict) -> None:
    """Classification affinée avec graphe."""
    notes = entry.notes or []

    # Pytest
    if "tests/" in entry.file_path:
        entry.classification = "PYTEST_AUTO"
        notes += ["Test/fixture auto-découvert"]
        entry.notes = notes
        return

    # Décorateur
    if entry.decorator:
        dec_name = entry.decorator.split('.')[-1].strip('@')
        if any(wh in dec_name for wh in DECORATOR_WHITELIST):
            entry.classification = "DECORATED_RUNTIME"
            notes += [f"Décorateur '{entry.decorator}' → usage runtime"]
            entry.notes = notes
            return

    # Graphe depviz
    func_id = f"fn:{entry.file_path}#{entry.function_name}"
    if any(e.get('to') == func_id for e in dep_graph.get('edges', [])):
        entry.classification = "GRAPH_CALLED"
        notes += ["Lien statique dans depviz.json"]
        entry.notes = notes
        return

    # Usages
    if entry.call_count > 0:
        entry.classification = "ACTUAL_CALLS"
    elif entry.usage_count > 0:
        entry.classification = "MENTIONED"
    else:
        entry.classification = "LIKELY_DEAD"
        notes += ["Aucun usage détecté"]

    entry.notes = notes

def audit_dead_functions_v4(entries: List[DeadFunctionEntry]) -> List[DeadFunctionEntry]:
    """Audit complet avec stats modulaires."""
    dep_graph = load_dep_graph()
    module_stats = defaultdict(lambda: {'total': 0, 'dead': 0})

    for entry in entries:
        find_definition_ast(entry)
        search_usages_ast(entry)
        classify_entry_v4(entry, dep_graph)

        # Stats module (insight)
        mod = '/'.join(entry.file_path.split('/')[:2])  # e.g., 'arz_model/numerics'
        module_stats[mod]['total'] += 1
        if entry.classification == "LIKELY_DEAD":
            module_stats[mod]['dead'] += 1
        entry.module_stats = {'total': module_stats[mod]['total'], 'dead': module_stats[mod]['dead']}

    return sorted(entries, key=lambda e: (-e.call_count, e.file_path))  # Priorise par usages

def summarize_v4(entries: List[DeadFunctionEntry]) -> Dict:
    """Insights : Counts + % par module."""
    counts = defaultdict(int)
    for e in entries:
        counts[e.classification] += 1

    mod_insights = {}
    for e in entries:
        mod = '/'.join(e.file_path.split('/')[:2])
        dead_pct = (e.module_stats['dead'] / e.module_stats['total'] * 100) if e.module_stats['total'] > 0 else 0
        mod_insights[mod] = {'dead_pct': round(dead_pct, 1), 'count': e.module_stats['dead']}

    return {'classifications': dict(counts), 'module_insights': dict(mod_insights)}

def write_outputs_v4(entries: List[DeadFunctionEntry]) -> None:
    """Outputs formés : JSON, MD tables, CSV."""
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_v4(entries)

    # JSON enrichi
    data = {'entries': [e.to_dict() for e in entries], 'summary': summary}
    (OUTPUT_BASE.with_suffix('.json')).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    # MD avec tables
    lines = ["# Dead Code Audit V4 - 2025-11-13", "", "## Summary", ""]
    # Table classifications
    lines += ["### Classifications"]
    lines += ["| Catégorie | Nombre |"]
    lines += ["|-----------|--------|"]
    for cat, cnt in sorted(summary['classifications'].items(), key=lambda x: x[1], reverse=True):
        lines += [f"| {cat} | {cnt} |"]
    lines += [""]

    # Table modules
    lines += ["### % Dead par Module (Priorisé)"]
    lines += ["| Module | Dead | % Dead |"]
    lines += ["|--------|------|--------|"]
    for mod, info in sorted(summary['module_insights'].items(), key=lambda x: x[1]['dead_pct'], reverse=True):
        lines += [f"| {mod} | {info['count']} | {info['dead_pct']}% |"]
    lines += [""]

    # Détails (limités)
    lines += ["## Détails Priorisés (par Usages)"]
    for entry in entries[:20]:  # Top 20
        notes = '; '.join(entry.notes or [])
        lines += [f"- **{entry.file_path}::{entry.function_name}** | {entry.classification} | Appels: {entry.call_count} | {notes}"]
    lines += ["*(... voir JSON pour complet)*"]

    (OUTPUT_BASE.with_suffix('.md')).write_text('\n'.join(lines), encoding='utf-8')

    # CSV pour Excel
    with (OUTPUT_BASE.with_suffix('.csv')).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file_path', 'function_name', 'classification', 'call_count', 'usage_count', 'notes'])
        writer.writeheader()
        for e in entries:
            writer.writerow({
                'file_path': e.file_path,
                'function_name': e.function_name,
                'classification': e.classification,
                'call_count': e.call_count,
                'usage_count': e.usage_count,
                'notes': '; '.join(e.notes or [])
            })

def main_v4() -> int:
    if not ANALYSIS_FILE.exists():
        print(f"Erreur : {ANALYSIS_FILE} manquant. Utilise architecture_analysis_v3.txt.", file=sys.stderr)
        return 1

    entries = parse_analysis_file()
    if not entries:
        print("Aucune entrée parsée – vérifie le format de l'analyse.txt.")
        return 1

    audited = audit_dead_functions_v4(entries)
    write_outputs_v4(audited)

    print(f"✅ Audit V4 complet ! ({len(audited)} entrées)")
    print(f"Rapports sauvés :")
    print(f"  - JSON : {OUTPUT_BASE}.json")
    print(f"  - MD (tables) : {OUTPUT_BASE}.md")
    print(f"  - CSV (Excel) : {OUTPUT_BASE}.csv")
    print("\nInsights rapides :")
    sumry = summarize_v4(audited)
    for cat, cnt in list(sumry['classifications'].items())[:5]:
        print(f"  - {cat}: {cnt}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main_v4())