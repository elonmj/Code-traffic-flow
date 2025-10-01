#!/usr/bin/env python3
"""
Script Maître - Orchestration de Tous les Tests de Validation Chapitre 7
Exécute séquentiellement tous les tests et génère le rapport de synthèse
"""

import sys
import subprocess
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import argparse

# Ajout du chemin vers les utilitaires
sys.path.append(str(Path(__file__).parent))
from validation_utils import create_summary_table, generate_tex_snippet

class ValidationOrchestrator:
    """Orchestrateur principal pour tous les tests de validation"""
    
    def __init__(self, output_dir="validation_ch7/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_scripts = [
            {
                "name": "Section 7.3 - Tests Analytiques",
                "script": "test_section_7_3_analytical.py",
                "revendications": ["R1", "R3"],
                "description": "Validation solutions analytiques et convergence numérique"
            },
            {
                "name": "Section 7.4 - Calibration Données Réelles", 
                "script": "test_section_7_4_calibration.py",
                "revendications": ["R2"],
                "description": "Calibration et validation sur données Victoria Island"
            },
            {
                "name": "Section 7.5 - Jumeau Numérique",
                "script": "test_section_7_5_digital_twin.py", 
                "revendications": ["R3", "R4", "R6"],
                "description": "Validation comportementale et conservation propriétés"
            },
            {
                "name": "Section 7.6 - Performance RL",
                "script": "test_section_7_6_rl_performance.py",
                "revendications": ["R5"], 
                "description": "Validation couplage ARZ-RL et performance agents"
            },
            {
                "name": "Section 7.7 - Tests Robustesse",
                "script": "test_section_7_7_robustness.py",
                "revendications": ["R6"],
                "description": "Tests conditions extrêmes et stabilité numérique"
            }
        ]
        
        # Critères de validation pour chaque revendication
        self.validation_criteria = {
            "R1": {
                "description": "Précision méthodes numériques",
                "metrics": ["convergence_order", "l2_error"],
                "thresholds": {"convergence_order": 4.5, "l2_error": 1e-3}
            },
            "R2": {
                "description": "Reproduction données observées", 
                "metrics": ["mape", "geh_acceptance"],
                "thresholds": {"mape": 15.0, "geh_acceptance": 85.0}
            },
            "R3": {
                "description": "Conservation propriétés physiques",
                "metrics": ["mass_conservation_error"],
                "thresholds": {"mass_conservation_error": 1e-6}
            },
            "R4": {
                "description": "Reproduction comportements observés",
                "metrics": ["behavioral_correlation", "temporal_accuracy"],
                "thresholds": {"behavioral_correlation": 0.8, "temporal_accuracy": 0.75}
            },
            "R5": {
                "description": "Performance agents RL",
                "metrics": ["travel_time_improvement", "coupling_stability"],
                "thresholds": {"travel_time_improvement": 5.0, "coupling_stability": 0.99}
            },
            "R6": {
                "description": "Robustesse conditions dégradées",
                "metrics": ["degraded_performance", "numerical_stability"],
                "thresholds": {"degraded_performance": 75.0, "numerical_stability": 0.95}
            }
        }
    
    def run_single_test(self, test_config, verbose=True):
        """Exécute un test individuel"""
        script_path = Path(__file__).parent / test_config["script"]
        
        if not script_path.exists():
            print(f"❌ Script non trouvé : {test_config['script']}")
            return {
                "test_name": test_config["name"],
                "status": "ERROR",
                "error": f"Script file not found: {test_config['script']}",
                "revendications": test_config["revendications"]
            }
        
        try:
            if verbose:
                print(f"\n{'='*60}")
                print(f"🔄 Exécution : {test_config['name']}")
                print(f"Script : {test_config['script']}")
                print(f"Revendications : {', '.join(test_config['revendications'])}")
                print(f"{'='*60}")
            
            # Exécution du script dans le bon répertoire de travail
            work_dir = Path(__file__).parent.parent.parent  # Code project directory
            result = subprocess.run(
                [sys.executable, str(script_path.relative_to(work_dir))],
                capture_output=True,
                text=True,
                cwd=work_dir
            )
            
            if result.returncode == 0:
                print(f"✅ {test_config['name']} : SUCCÈS")
                return {
                    "test_name": test_config["name"],
                    "status": "SUCCESS",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "revendications": test_config["revendications"]
                }
            else:
                print(f"❌ {test_config['name']} : ÉCHEC (code {result.returncode})")
                if result.stderr:
                    print(f"🔍 Erreur stderr: {result.stderr[:500]}")
                if result.stdout:
                    print(f"🔍 Sortie stdout: {result.stdout[-500:]}")
                return {
                    "test_name": test_config["name"],
                    "status": "FAILED", 
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "revendications": test_config["revendications"]
                }
                
        except Exception as e:
            print(f"❌ {test_config['name']} : ERREUR - {e}")
            return {
                "test_name": test_config["name"],
                "status": "ERROR",
                "error": str(e),
                "revendications": test_config["revendications"]
            }

    
    def analyze_revendications_status(self, test_results):
        """Analyse le statut de validation de chaque revendication"""
        revendications_status = {}
        
        for rev_id, criteria in self.validation_criteria.items():
            revendications_status[rev_id] = {
                "description": criteria["description"],
                "status": "NON_TESTED",
                "tests_passed": 0,
                "tests_total": 0,
                "details": []
            }
        
        # Analyse des résultats de tests
        for result in test_results:
            if result["status"] == "SUCCESS":
                for rev_id in result["revendications"]:
                    if rev_id in revendications_status:
                        revendications_status[rev_id]["tests_passed"] += 1
                        revendications_status[rev_id]["tests_total"] += 1
                        revendications_status[rev_id]["details"].append(f"✅ {result['test_name']}")
            else:
                for rev_id in result["revendications"]:
                    if rev_id in revendications_status:
                        revendications_status[rev_id]["tests_total"] += 1
                        revendications_status[rev_id]["details"].append(f"❌ {result['test_name']}")
        
        # Détermination du statut final
        for rev_id, status in revendications_status.items():
            if status["tests_total"] == 0:
                status["status"] = "NON_TESTED"
            elif status["tests_passed"] == status["tests_total"]:
                status["status"] = "VALIDATED"
            elif status["tests_passed"] > 0:
                status["status"] = "PARTIAL"
            else:
                status["status"] = "FAILED"
        
        return revendications_status
    
    def generate_executive_summary(self, test_results, revendications_status):
        """Génère le résumé exécutif de la validation"""
        
        # Statistiques globales
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results if r["status"] == "SUCCESS")
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Statut des revendications
        validated_revs = sum(1 for r in revendications_status.values() if r["status"] == "VALIDATED")
        total_revs = len(revendications_status)
        validation_rate = (validated_revs / total_revs) * 100 if total_revs > 0 else 0
        
        summary = {
            "execution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "validated_revendications": validated_revs,
            "total_revendications": total_revs,
            "validation_rate": validation_rate,
            "overall_status": "VALIDATED" if validation_rate >= 83.33 else "PARTIAL"  # 5/6 revendications
        }
        
        return summary
    
    def generate_latex_synthesis(self, test_results, revendications_status, executive_summary):
        """Génère la synthèse LaTeX finale"""
        
        # Template pour tableau de synthèse des revendications
        synthesis_template = """
% Synthèse Validation Chapitre 7 - Auto-generated
% Date: {execution_date}

\\subsection{{Synthèse de la Validation}}

Cette section présente la synthèse complète de la validation des six revendications du Chapitre 7.

\\subsubsection{{Résumé Exécutif}}

\\begin{{itemize}}
    \\item \\textbf{{Tests exécutés}} : {successful_tests}/{total_tests} ({success_rate:.1f}\\% de succès)
    \\item \\textbf{{Revendications validées}} : {validated_revendications}/{total_revendications} ({validation_rate:.1f}\\% de validation)
    \\item \\textbf{{Statut global}} : \\textbf{{{overall_status_text}}}
\\end{{itemize}}

\\subsubsection{{Statut des Revendications}}

\\begin{{table}}[H]
\\centering
\\caption{{Synthèse de validation des revendications}}
\\label{{tab:revendications_synthesis}}
\\begin{{tabular}}{{|l|l|c|c|c|}}
\\hline
\\textbf{{ID}} & \\textbf{{Description}} & \\textbf{{Tests}} & \\textbf{{Réussis}} & \\textbf{{Statut}} \\\\
\\hline
{revendications_rows}
\\hline
\\end{{tabular}}
\\end{{table}}

\\subsubsection{{Détail par Section}}

\\begin{{table}}[H]
\\centering
\\caption{{Résultats par section de validation}}
\\label{{tab:sections_results}}
\\begin{{tabular}}{{|l|c|c|l|}}
\\hline
\\textbf{{Section}} & \\textbf{{Statut}} & \\textbf{{Revendications}} & \\textbf{{Commentaires}} \\\\
\\hline
{sections_rows}
\\hline
\\end{{tabular}}
\\end{{table}}

\\textbf{{Conclusion}} : {conclusion_text}
"""
        
        # Génération des lignes du tableau des revendications
        revendications_rows = []
        for rev_id, status in revendications_status.items():
            status_symbol = {
                "VALIDATED": "✓",
                "PARTIAL": "⚠",
                "FAILED": "✗",
                "NON_TESTED": "◯"
            }.get(status["status"], "?")
            
            row = f"{rev_id} & {status['description']} & {status['tests_total']} & {status['tests_passed']} & {status_symbol} \\\\"
            revendications_rows.append(row)
        
        # Génération des lignes du tableau des sections
        sections_rows = []
        for result in test_results:
            status_symbol = "✓" if result["status"] == "SUCCESS" else "✗"
            revendications_list = ", ".join(result["revendications"])
            comment = "Validation réussie" if result["status"] == "SUCCESS" else "Nécessite attention"
            
            row = f"{result['test_name']} & {status_symbol} & {revendications_list} & {comment} \\\\"
            sections_rows.append(row)
        
        # Détermination du texte de conclusion
        if executive_summary["validation_rate"] >= 100:
            conclusion_text = "Toutes les revendications sont validées avec succès. Le modèle ARZ et le framework RL satisfont tous les critères de validation."
        elif executive_summary["validation_rate"] >= 83.33:
            conclusion_text = "La majorité des revendications sont validées. Les éléments non validés nécessitent des améliorations ciblées."
        else:
            conclusion_text = "La validation révèle des lacunes importantes nécessitant des révisions majeures du modèle et/ou des méthodes."
        
        # Formatage du template
        template_data = executive_summary.copy()
        template_data.update({
            "revendications_rows": "\n".join(revendications_rows),
            "sections_rows": "\n".join(sections_rows),
            "overall_status_text": "VALIDÉ" if executive_summary["overall_status"] == "VALIDATED" else "PARTIEL",
            "conclusion_text": conclusion_text
        })
        
        # Génération du fichier
        synthesis_content = synthesis_template.format(**template_data)
        output_path = self.output_dir / "chapter_7_synthesis.tex"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(synthesis_content)
        
        print(f"📄 Synthèse LaTeX générée : {output_path}")
        return output_path
    
    def run_all_tests(self, sections=None, verbose=True):
        """Exécute tous les tests de validation"""
        
        print("🚀 DÉMARRAGE VALIDATION COMPLÈTE CHAPITRE 7")
        print(f"Répertoire de sortie : {self.output_dir}")
        print(f"Nombre de tests : {len(self.test_scripts)}")
        
        test_results = []
        
        # Exécution des tests sélectionnés
        for test_config in self.test_scripts:
            if sections is None or any(sec in test_config["name"] for sec in sections):
                result = self.run_single_test(test_config, verbose)
                test_results.append(result)
        
        # Analyse des revendications
        revendications_status = self.analyze_revendications_status(test_results)
        
        # Génération résumé exécutif
        executive_summary = self.generate_executive_summary(test_results, revendications_status)
        
        # Génération synthèse LaTeX
        synthesis_path = self.generate_latex_synthesis(test_results, revendications_status, executive_summary)
        
        # Sauvegarde des résultats complets
        full_results = {
            "executive_summary": executive_summary,
            "test_results": test_results,
            "revendications_status": revendications_status,
            "synthesis_file": str(synthesis_path)
        }
        
        results_json_path = self.output_dir / "full_validation_results.json"
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Affichage du résumé final
        self.print_final_summary(executive_summary, revendications_status)
        
        return full_results
    
    def print_final_summary(self, executive_summary, revendications_status):
        """Affiche le résumé final de la validation"""
        
        print(f"\n{'='*80}")
        print("🎯 RÉSUMÉ FINAL DE LA VALIDATION")
        print(f"{'='*80}")
        
        print(f"📊 Tests : {executive_summary['successful_tests']}/{executive_summary['total_tests']} réussis ({executive_summary['success_rate']:.1f}%)")
        print(f"🎯 Revendications : {executive_summary['validated_revendications']}/{executive_summary['total_revendications']} validées ({executive_summary['validation_rate']:.1f}%)")
        
        print(f"\n📋 DÉTAIL DES REVENDICATIONS :")
        for rev_id, status in revendications_status.items():
            status_icon = {
                "VALIDATED": "✅",
                "PARTIAL": "⚠️", 
                "FAILED": "❌",
                "NON_TESTED": "⭕"
            }.get(status["status"], "❓")
            
            print(f"  {status_icon} {rev_id}: {status['description']} ({status['tests_passed']}/{status['tests_total']} tests)")
        
        print(f"\n🏆 STATUT GLOBAL : {executive_summary['overall_status']}")
        
        if executive_summary['overall_status'] == 'VALIDATED':
            print("🎉 FÉLICITATIONS ! Toutes les validations sont réussies.")
        else:
            print("🔧 Des améliorations sont nécessaires sur certains aspects.")
        
        print(f"{'='*80}")

def main():
    """Fonction principale avec arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="Orchestrateur de validation Chapitre 7")
    parser.add_argument("--sections", nargs="*", help="Sections spécifiques à tester")
    parser.add_argument("--quiet", action="store_true", help="Mode silencieux")
    parser.add_argument("--output", default="validation_ch7/results", help="Répertoire de sortie")
    
    args = parser.parse_args()
    
    # Initialisation de l'orchestrateur
    orchestrator = ValidationOrchestrator(output_dir=args.output)
    
    # Exécution des tests
    results = orchestrator.run_all_tests(
        sections=args.sections,
        verbose=not args.quiet
    )
    
    return results

if __name__ == "__main__":
    main()