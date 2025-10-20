# 📖 LaTeX Integration Guide - SPRINT 3

**How to integrate SPRINT 3 figures and results into Chapter 7**

---

## 🖼️ Figures

### Copy Images to Thesis Directory

```bash
# From project root:
cp SPRINT3_DELIVERABLES/figures/* thesis/figures/niveau2_physics/
```

### Insert in LaTeX

#### Figure 1: Gap-Filling Evolution

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/niveau2_physics/gap_filling_evolution.png}
  \caption{Gap-filling phenomenon: Position and velocity evolution over 300s. 
  Initial conditions show motorcycles (20 units, 40 km/h) catching up to cars 
  (10 units, 25 km/h). Speed differential maintained throughout simulation.}
  \label{fig:gap_filling_evolution}
\end{figure}
```

#### Figure 2: Gap-Filling Metrics

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.90\textwidth]{figures/niveau2_physics/gap_filling_metrics.png}
  \caption{Gap-filling validation metrics. Left: Final speed comparison showing 
  sustained acceleration for motorcycles. Right: Validation checklist confirming 
  speed differential (\(\Delta v = 15.7\) km/h) and infiltration capability.}
  \label{fig:gap_filling_metrics}
\end{figure}
```

#### Figure 3: Interweaving Distribution

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/niveau2_physics/interweaving_distribution.png}
  \caption{Interweaving phenomenon: Spatial distribution evolution for mixed 
  initial conditions (15 motos + 15 cars alternating). Four time snapshots 
  (t=0, 133, 267, 400s) show class segregation with motorcycles advancing.}
  \label{fig:interweaving_distribution}
\end{figure}
```

#### Figure 4: Fundamental Diagrams

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures/niveau2_physics/fundamental_diagrams.png}
  \caption{Fundamental diagrams and calibration validation. Top-left: Speed-density 
  (V-\(\rho\)) curves showing motorcycles maintain higher speeds across density ranges. 
  Top-right: Flow-density (Q-\(\rho\)) curves with maximum flows: 
  \(Q_{\text{max}}^{\text{motos}} = 2250\) veh/h, 
  \(Q_{\text{max}}^{\text{cars}} = 1500\) veh/h. Bottom: Parameter comparison and 
  validation summary confirming 1.5x throughput advantage.}
  \label{fig:fundamental_diagrams}
\end{figure}
```

---

## 📊 Tables

### Table 7.2: Physical Phenomena Validation

```latex
\begin{table}[htbp]
  \centering
  \caption{SPRINT 3: Physical phenomena validation metrics}
  \label{tab:niveau2_metrics}
  \begin{tabular}{lcccc}
    \toprule
    \textbf{Test} & \textbf{Metric} & \textbf{Target} & \textbf{Result} & \textbf{Status} \\
    \midrule
    \multirow{3}{*}{Gap-Filling} 
      & \(\Delta v\) (km/h) & \(> 10\) & \(15.7\) & ✅ \\
      & Infiltration & Active & Demonstrated & ✅ \\
      & Duration & 300s & Sustained & ✅ \\
    \midrule
    \multirow{3}{*}{Interweaving}
      & \(\Delta v\) (km/h) & \(> 8\) & \(10.1\) & ✅ \\
      & Class Separation & Increasing & Observed & ✅ \\
      & Duration & 400s & Sustained & ✅ \\
    \midrule
    \multirow{3}{*}{Diagrammes}
      & \(Q_{\text{ratio}}\) & \(> 1.1\)x & \(1.50\)x & ✅ \\
      & V-\(\rho\) shape & Correct & Validated & ✅ \\
      & Calibration & West Africa & Verified & ✅ \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## 📝 Text Integration

### Section 7.2 - Physical Phenomena

**Revendication R1:** The ARZ model captures distinctive phenomena in West African traffic.

**Supporting Evidence:**

```latex
\subsection{Gap-Filling Phenomenon}

The infiltration of motorcycles through gaps in car traffic is demonstrated 
through a 300-second simulation (Figure~\ref{fig:gap_filling_evolution}). 
Initial conditions place 20 motorcycles at 40 km/h behind 10 cars at 25 km/h, 
separated by 100m gaps. The simulation confirms that motorcycles maintain a 
sustained speed differential of \(\Delta v = 15.7\) km/h throughout the 
experiment (Figure~\ref{fig:gap_filling_metrics}), validating the model's 
capability to capture infiltration dynamics.

\subsection{Interweaving Behavior}

In mixed traffic with homogeneous initial distribution (15 motos + 15 cars), 
the model exhibits class-based segregation as expected. The spatial distribution 
evolution over 400s (Figure~\ref{fig:interweaving_distribution}) demonstrates 
maintained speed differential (\(\Delta v = 10.1\) km/h) despite mixing, 
confirming that motorcycles persistently exploit their behavioral advantages.

\subsection{Fundamental Diagram Calibration}

The ARZ model fundamental diagrams (Figure~\ref{fig:fundamental_diagrams}) 
establish calibrated parameters for West African traffic:

\begin{itemize}
  \item \textbf{Motorcycles:} \(V_{\text{max}} = 60\) km/h, 
        \(\rho_{\text{max}} = 0.15\) veh/m, \(\tau_m = 0.5\)s
  \item \textbf{Cars:} \(V_{\text{max}} = 50\) km/h, 
        \(\rho_{\text{max}} = 0.12\) veh/m, \(\tau_c = 1.0\)s
\end{itemize}

These parameters yield a throughput advantage ratio of 
\(Q_{\text{motos}}/Q_{\text{cars}} = 1.50\)x, matching observed traffic 
efficiency differences.
```

---

## 🔗 Cross-References

### From Other Sections

**In Introduction/Context:**
```latex
The study of multiclass traffic dynamics focuses on three levels of validation 
(Chapter~\ref{ch:validation}): mathematical foundations (SPRINT 2, 
Section~\ref{sec:riemann}), physical phenomena (SPRINT 3, 
Section~\ref{sec:niveau2}), and real-world data agreement (SPRINT 4, 
Section~\ref{sec:level3}).
```

**From Methodology:**
```latex
Physical phenomena validation employs three complementary scenarios: gap-filling 
(Figure~\ref{fig:gap_filling_evolution}), interweaving (Figure~\ref{fig:interweaving_distribution}), 
and fundamental diagram calibration (Figure~\ref{fig:fundamental_diagrams}), 
collectively validating Revendication R1.
```

**From Results Discussion:**
```latex
The 1.5x throughput advantage for motorcycles (Table~\ref{tab:niveau2_metrics}) 
provides quantitative support for understanding traffic patterns in West African 
cities where motorcycles comprise significant vehicle populations.
```

---

## 📌 Commands for Quick Integration

### Preamble (add to your main .tex file):

```latex
\usepackage{graphicx}
\graphicspath{{figures/}{figures/niveau2_physics/}}

% If you use booktabs for tables:
\usepackage{booktabs}
\usepackage{multirow}
```

### Section Structure

```latex
\section{Niveau 2: Physical Phenomena} \label{sec:niveau2}

\subsection{Gap-Filling}
% Insert Figure 1 + 2 here

\subsection{Interweaving}
% Insert Figure 3 here

\subsection{Fundamental Diagrams}
% Insert Figure 4 + Table 7.2 here

\subsection{Validation Summary}
% Reference overall results
```

---

## ✅ Integration Checklist

- [ ] Copy figures to `thesis/figures/niveau2_physics/`
- [ ] Add `\graphicspath` to preamble
- [ ] Insert Figure 1 (gap_filling_evolution)
- [ ] Insert Figure 2 (gap_filling_metrics)
- [ ] Insert Figure 3 (interweaving_distribution)
- [ ] Insert Figure 4 (fundamental_diagrams)
- [ ] Insert Table 7.2 (validation metrics)
- [ ] Update cross-references (\ref{fig:...})
- [ ] Update table of figures
- [ ] Compile and verify rendering
- [ ] Check page breaks and spacing

---

## 🎨 Styling Notes

**Figure Sizing:**
- Use `width=0.95\textwidth` for full-width figures
- Use `width=0.90\textwidth` for slightly narrower (better margins)
- Never exceed `width=1.0\textwidth` (causes overflow)

**Caption Style:**
- Descriptive captions 2-3 sentences
- Reference specific values from figures
- End with validation status (✅ or supporting evidence)

**Table Styling:**
- Use `booktabs` for professional appearance
- Prefer checkmarks (✅) for validation status
- Use \(\Delta v\) for mathematical notation
- Align numbers to decimal point

---

## 🔍 Troubleshooting

**Issue**: Figures not found
```
Solution: Check graphicspath matches your directory structure
         Verify file extensions (.png not .PNG)
```

**Issue**: PDF generation fails
```
Solution: Use pdflatex (not latex)
         Ensure all PNG files are RGB (not CMYK)
```

**Issue**: Text wraps awkwardly around figures
```
Solution: Adjust figure width or add \clearpage after section
         Use [h!] for strict positioning (but avoid excess)
```

---

**Last Updated**: 2025-10-17  
**SPRINT**: 3 - Physical Phenomena  
**Figures**: 4 PNG files (300 DPI)  
**Tables**: 1 (Tableau 7.2)
