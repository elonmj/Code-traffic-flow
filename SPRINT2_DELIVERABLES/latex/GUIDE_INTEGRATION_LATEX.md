# 📘 GUIDE D'INTÉGRATION LATEX - SPRINT 2

## 🎯 Utilisation Rapide

### Dans votre fichier LaTeX principal

```latex
% Dans le préambule
\usepackage{graphicx}
\usepackage{booktabs}  % Pour les tableaux

% Dans le chapitre 7 (Validation)

% 1. Inclure le tableau de résultats
\input{SPRINT2_DELIVERABLES/latex/table71_updated.tex}

% 2. Inclure les figures
\input{SPRINT2_DELIVERABLES/latex/figures_integration.tex}
```

---

## 📊 Fichiers LaTeX Disponibles

### 1. `table71_updated.tex`
**Tableau 7.1** avec tous les résultats de validation (5 tests + convergence)

**Contenu:**
- Erreurs L2 pour les 5 tests de Riemann
- Ordre de convergence moyen (4.78)
- Notes explicatives sur les maillages et critères
- Statuts de validation ✅

**Utilisation:**
```latex
\section{Résultats de validation}
\input{SPRINT2_DELIVERABLES/latex/table71_updated.tex}
```

---

### 2. `figures_integration.tex`
**6 figures PNG** (300 DPI) avec captions complètes

**Figures incluses:**
1. `test1_shock_motos.png` - Choc simple motos (2 subplots)
2. `test2_rarefaction_motos.png` - Détente simple motos (2 subplots)
3. `test3_shock_voitures.png` - Choc simple voitures
4. `test4_rarefaction_voitures.png` - Détente simple voitures
5. **`test5_multiclass_interaction.png`** ⭐ - Interaction multiclasse (3 subplots - CRITIQUE)
6. `convergence_study_weno5.png` - Étude de convergence

**Utilisation:**
```latex
\section{Validation graphique}
\input{SPRINT2_DELIVERABLES/latex/figures_integration.tex}
```

---

## 🏷️ Labels pour Références Croisées

Utilisez ces labels dans votre texte :

```latex
% Références aux figures
Comme montré dans la Figure~\ref{fig:riemann_choc_motos}, ...
La Figure~\ref{fig:riemann_multiclass_interaction} valide notre contribution centrale.
L'étude de convergence (Figure~\ref{fig:convergence_weno5}) confirme l'ordre WENO5.

% Référence au tableau
Le Tableau~\ref{tab:validation_riemann_niveau1} récapitule tous les résultats.
```

**Labels disponibles:**
- `fig:riemann_choc_motos` - Test 1
- `fig:riemann_rarefaction_motos` - Test 2
- `fig:riemann_choc_voitures` - Test 3
- `fig:riemann_rarefaction_voitures` - Test 4
- `fig:riemann_multiclass_interaction` - Test 5 ⭐ CRITIQUE
- `fig:convergence_weno5` - Convergence
- `tab:validation_riemann_niveau1` - Tableau résultats

---

## 📁 Structure des Fichiers

```
SPRINT2_DELIVERABLES/
├── figures/
│   ├── test1_shock_motos.png (241.8 KB)
│   ├── test2_rarefaction_motos.png (578.5 KB)
│   ├── test3_shock_voitures.png (118.7 KB)
│   ├── test4_rarefaction_voitures.png (122.0 KB)
│   ├── test5_multiclass_interaction.png (363.4 KB) ⭐
│   └── convergence_study_weno5.png (232.4 KB)
│
└── latex/
    ├── table71_updated.tex
    └── figures_integration.tex (ce fichier inclut toutes les figures)
```

---

## 💡 Exemple Complet - Chapitre 7

```latex
\chapter{Validation du Jumeau Numérique}

\section{Introduction}
Ce chapitre présente la validation complète du jumeau numérique...

\section{Niveau 1 : Fondations Mathématiques}

\subsection{Tests de Riemann}
Nous avons implémenté 5 tests de Riemann pour valider la précision 
du schéma numérique (voir Tableau~\ref{tab:validation_riemann_niveau1}).

% Inclure le tableau
\input{SPRINT2_DELIVERABLES/latex/table71_updated.tex}

\subsection{Résultats Graphiques}
Les Figures~\ref{fig:riemann_choc_motos} à \ref{fig:convergence_weno5} 
présentent les profils de densité et vitesse pour chaque test.

% Inclure toutes les figures
\input{SPRINT2_DELIVERABLES/latex/figures_integration.tex}

\subsection{Discussion}
Le Test 5 (Figure~\ref{fig:riemann_multiclass_interaction}) est particulièrement 
critique car il valide notre contribution centrale : le couplage faible qui 
maintient le différentiel de vitesse entre motos et voitures...
```

---

## 🔧 Personnalisation

### Modifier la largeur des figures

Dans `figures_integration.tex`, changez `width`:
```latex
% Plus petite
\includegraphics[width=0.7\textwidth]{...}

% Plus grande
\includegraphics[width=1.0\textwidth]{...}

% Largeur fixe
\includegraphics[width=12cm]{...}
```

### Modifier les captions

Éditez directement `figures_integration.tex` pour ajuster le texte des légendes.

### Ajouter des sous-figures

Pour organiser plusieurs tests ensemble :
```latex
\begin{figure}[h!]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{test1_shock_motos.png}
    \caption{Test 1}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{test2_rarefaction_motos.png}
    \caption{Test 2}
\end{subfigure}
\caption{Comparaison choc vs détente}
\end{figure}
```

---

## ✅ Checklist d'Intégration

- [ ] Copier `SPRINT2_DELIVERABLES/` dans le dossier de votre thèse
- [ ] Vérifier que `\usepackage{graphicx}` et `\usepackage{booktabs}` sont dans le préambule
- [ ] Inclure `table71_updated.tex` dans le chapitre 7
- [ ] Inclure `figures_integration.tex` dans le chapitre 7
- [ ] Compiler avec `pdflatex` (ou `xelatex`)
- [ ] Vérifier les références croisées (re-compiler si nécessaire)
- [ ] Vérifier la qualité des figures dans le PDF final

---

## 🚨 Résolution de Problèmes

### Figures ne s'affichent pas
```latex
% Vérifier le chemin relatif
\graphicspath{{./SPRINT2_DELIVERABLES/figures/}}

% Ou utiliser des chemins absolus temporairement
\includegraphics[width=0.85\textwidth]{/chemin/complet/vers/test1_shock_motos.png}
```

### Tableau trop large
```latex
% Réduire la taille
\begin{table}[h!]
\centering
\small  % ou \footnotesize
\input{SPRINT2_DELIVERABLES/latex/table71_updated.tex}
\end{table}
```

### Compilation lente
Les PNG sont optimisés (300 DPI), mais si compilation trop lente :
```latex
% Mode draft (bounding boxes seulement)
\documentclass[draft]{report}
```

---

## 📞 Support

Pour toute question sur l'intégration LaTeX :
1. Consulter `README.md` dans `SPRINT2_DELIVERABLES/`
2. Vérifier `EXECUTIVE_SUMMARY.md` pour le contexte complet
3. Consulter `code/CODE_INDEX.md` pour les scripts sources

---

**Créé le:** 17 octobre 2025  
**Format:** PNG 300 DPI (compatible LaTeX)  
**Encodage:** UTF-8
