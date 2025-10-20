#!/usr/bin/env python3
"""
Script de validation: Vérifier que run_section_7_6.py est conforme à test_section_7_6_rl_performance.py

Ce script vérifie l'architecture SANS exécuter le training (pas d'imports lourds).
"""

import sys
from pathlib import Path
import ast
import inspect

# Colors pour terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def check_file_exists(filepath: Path, name: str) -> bool:
    """Vérifie qu'un fichier existe."""
    exists = filepath.exists()
    status = f"{GREEN}✅{RESET}" if exists else f"{RED}❌{RESET}"
    print(f"  {status} {name}: {filepath.name}")
    if not exists:
        print(f"      {RED}Fichier manquant!{RESET}")
    return exists

def check_function_exists_in_file(filepath: Path, function_name: str) -> bool:
    """Vérifie qu'une fonction existe dans un fichier Python."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        exists = function_name in functions
        status = f"{GREEN}✅{RESET}" if exists else f"{RED}❌{RESET}"
        print(f"    {status} Fonction: {function_name}()")
        return exists
    except Exception as e:
        print(f"    {RED}❌{RESET} Erreur lecture: {e}")
        return False

def check_class_exists_in_file(filepath: Path, class_name: str) -> bool:
    """Vérifie qu'une classe existe dans un fichier Python."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        exists = class_name in classes
        status = f"{GREEN}✅{RESET}" if exists else f"{RED}❌{RESET}"
        print(f"    {status} Classe: {class_name}")
        return exists
    except Exception as e:
        print(f"    {RED}❌{RESET} Erreur lecture: {e}")
        return False

def check_import_statement(filepath: Path, import_from: str, import_name: str) -> bool:
    """Vérifie qu'un import existe dans un fichier."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check: from X import Y
        exists = f"from {import_from} import {import_name}" in content
        status = f"{GREEN}✅{RESET}" if exists else f"{RED}❌{RESET}"
        print(f"    {status} Import: from {import_from} import {import_name}")
        return exists
    except Exception as e:
        print(f"    {RED}❌{RESET} Erreur lecture: {e}")
        return False

def main():
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}VALIDATION ARCHITECTURE: run_section_7_6.py vs test_section_7_6_rl_performance.py{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    
    base_dir = Path(__file__).parent
    
    results = []
    
    # ========================================================================
    # 1. VÉRIFIER EXISTENCE FICHIERS
    # ========================================================================
    print(f"{BOLD}1. VÉRIFICATION EXISTENCE FICHIERS{RESET}")
    print("-" * 80)
    
    run_file = base_dir / "run_section_7_6.py"
    rl_training_file = base_dir / "rl_training.py"
    rl_evaluation_file = base_dir / "rl_evaluation.py"
    readme_file = base_dir / "README_SECTION_7_6.md"
    
    results.append(check_file_exists(run_file, "Script final unique"))
    results.append(check_file_exists(rl_training_file, "Module RL Training"))
    results.append(check_file_exists(rl_evaluation_file, "Module RL Evaluation"))
    results.append(check_file_exists(readme_file, "Documentation README"))
    
    # ========================================================================
    # 2. VÉRIFIER IMPORTS RÉELS (PAS DE MOCK)
    # ========================================================================
    print(f"\n{BOLD}2. VÉRIFICATION IMPORTS RÉELS (PAS DE MOCK){RESET}")
    print("-" * 80)
    
    print(f"\n  {BOLD}run_section_7_6.py:{RESET}")
    results.append(check_import_statement(run_file, "rl_training", "train_rl_agent_for_validation"))
    results.append(check_import_statement(run_file, "rl_evaluation", "evaluate_traffic_performance"))
    
    print(f"\n  {BOLD}rl_training.py:{RESET}")
    results.append(check_import_statement(rl_training_file, "Code_RL.src.env.traffic_signal_env_direct", "TrafficSignalEnvDirect"))
    results.append(check_import_statement(rl_training_file, "stable_baselines3", "DQN"))
    
    print(f"\n  {BOLD}rl_evaluation.py:{RESET}")
    results.append(check_import_statement(rl_evaluation_file, "Code_RL.src.env.traffic_signal_env_direct", "TrafficSignalEnvDirect"))
    results.append(check_import_statement(rl_evaluation_file, "stable_baselines3", "DQN"))
    
    # ========================================================================
    # 3. VÉRIFIER CLASSES CRITIQUES
    # ========================================================================
    print(f"\n{BOLD}3. VÉRIFICATION CLASSES CRITIQUES{RESET}")
    print("-" * 80)
    
    print(f"\n  {BOLD}run_section_7_6.py:{RESET}")
    results.append(check_class_exists_in_file(run_file, "Section76Config"))
    results.append(check_class_exists_in_file(run_file, "Section76Orchestrator"))
    
    print(f"\n  {BOLD}rl_training.py:{RESET}")
    results.append(check_class_exists_in_file(rl_training_file, "RLTrainer"))
    
    print(f"\n  {BOLD}rl_evaluation.py:{RESET}")
    results.append(check_class_exists_in_file(rl_evaluation_file, "BaselineController"))
    results.append(check_class_exists_in_file(rl_evaluation_file, "RLController"))
    results.append(check_class_exists_in_file(rl_evaluation_file, "TrafficEvaluator"))
    
    # ========================================================================
    # 4. VÉRIFIER FONCTIONS CRITIQUES
    # ========================================================================
    print(f"\n{BOLD}4. VÉRIFICATION FONCTIONS CRITIQUES{RESET}")
    print("-" * 80)
    
    print(f"\n  {BOLD}run_section_7_6.py:{RESET}")
    results.append(check_function_exists_in_file(run_file, "phase_1_train_rl_agent"))
    results.append(check_function_exists_in_file(run_file, "phase_2_evaluate_strategies"))
    results.append(check_function_exists_in_file(run_file, "phase_3_generate_outputs"))
    results.append(check_function_exists_in_file(run_file, "_generate_performance_comparison_figure"))
    results.append(check_function_exists_in_file(run_file, "_generate_learning_curve"))
    results.append(check_function_exists_in_file(run_file, "_generate_latex_performance_table"))
    results.append(check_function_exists_in_file(run_file, "_generate_latex_content"))
    
    print(f"\n  {BOLD}rl_training.py:{RESET}")
    results.append(check_function_exists_in_file(rl_training_file, "train_rl_agent_for_validation"))
    
    print(f"\n  {BOLD}rl_evaluation.py:{RESET}")
    results.append(check_function_exists_in_file(rl_evaluation_file, "evaluate_traffic_performance"))
    
    # ========================================================================
    # 5. VÉRIFIER CONTENU CODE_RL_HYPERPARAMETERS
    # ========================================================================
    print(f"\n{BOLD}5. VÉRIFICATION CODE_RL_HYPERPARAMETERS{RESET}")
    print("-" * 80)
    
    try:
        with open(rl_training_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        critical_params = [
            ("learning_rate", "1e-3"),
            ("batch_size", "32"),
            ("buffer_size", "50000"),
            ("gamma", "0.99")
        ]
        
        for param, expected_val in critical_params:
            found = f'"{param}": {expected_val}' in content or f"'{param}': {expected_val}" in content
            status = f"{GREEN}✅{RESET}" if found else f"{RED}❌{RESET}"
            print(f"  {status} {param} = {expected_val}")
            results.append(found)
    except Exception as e:
        print(f"  {RED}❌{RESET} Erreur lecture hyperparams: {e}")
        results.append(False)
    
    # ========================================================================
    # 6. VÉRIFIER MODE --QUICK
    # ========================================================================
    print(f"\n{BOLD}6. VÉRIFICATION MODE --QUICK INTÉGRÉ{RESET}")
    print("-" * 80)
    
    try:
        with open(run_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('QUICK_TIMESTEPS', 'QUICK_TIMESTEPS = 100'),
            ('QUICK_EPISODES', 'QUICK_EPISODES = 1'),
            ('CLI --quick', 'add_argument("--quick"'),
            ('quick_mode param', 'quick_mode: bool'),
        ]
        
        for name, pattern in checks:
            found = pattern in content
            status = f"{GREEN}✅{RESET}" if found else f"{RED}❌{RESET}"
            print(f"  {status} {name}")
            results.append(found)
    except Exception as e:
        print(f"  {RED}❌{RESET} Erreur vérification --quick: {e}")
        results.append(False)
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}RÉSUMÉ VALIDATION{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    
    total_checks = len(results)
    passed_checks = sum(results)
    failed_checks = total_checks - passed_checks
    
    print(f"  Total vérifications: {total_checks}")
    print(f"  {GREEN}✅ Réussies: {passed_checks}{RESET}")
    print(f"  {RED}❌ Échecs: {failed_checks}{RESET}")
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"\n  Taux de réussite: {success_rate:.1f}%")
    
    if success_rate == 100:
        print(f"\n  {GREEN}{BOLD}🎉 VALIDATION COMPLÈTE: Architecture 100% conforme!{RESET}")
        print(f"  {GREEN}✅ run_section_7_6.py est prêt pour exécution.{RESET}")
        return 0
    elif success_rate >= 90:
        print(f"\n  {YELLOW}{BOLD}⚠️  VALIDATION PRESQUE COMPLÈTE: Quelques ajustements mineurs nécessaires.{RESET}")
        return 1
    else:
        print(f"\n  {RED}{BOLD}❌ VALIDATION ÉCHOUÉE: Architecture non conforme.{RESET}")
        print(f"  {RED}Corriger les erreurs ci-dessus avant exécution.{RESET}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
