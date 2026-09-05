# R&D / AI Assignment: Parametric Curve Parameter Estimation

## Problem Statement

Find the values of unknown variables $\theta, M, X$ in the parametric equation:
$$x(t) = t \cos(\theta) - e^{M |t|} \sin(0.3 t) \sin(\theta) + X$$
$$y(t) = 42 + t \sin(\theta) + e^{M |t|} \sin(0.3 t) \cos(\theta)$$

Given constraints:
- $0^\circ < \theta < 50^\circ$ ($0 < \theta < 0.8726$ rad)
- $-0.05 < M < 0.05$
- $0 < X < 100$
- $6 < t < 60$
- Empirical point set: `xy_data.csv` (1,500 points)

---

## 1. Solution & Optimal Parameters

Through orthogonal decomposition and multi-start L-BFGS-B optimization (`solve_curve.py`):

| Variable | Optimal Numerical Value | Exact Analytical Value |
|---|---|---|
| **$\theta$** | **0.523598 rad** | **$30^\circ = \frac{\pi}{6}$ rad** |
| **$M$** | **0.030000** | **0.03** |
| **$X$** | **54.999998** | **55.0** |

**Empirical L1 Fit Error:** $\approx 3.5 \times 10^{-6}$ (essentially zero error across all 1,500 points).

---

## 2. Desmos Graph & LaTeX Submission

### Desmos Calculator Link
The template calculator provided in the assignment is:
👉 **[https://www.desmos.com/calculator/rfj91yrxob](https://www.desmos.com/calculator/rfj91yrxob)**

### Parametric Equation in LaTeX Format (for Desmos / README)
```latex
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,\ 42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```
*(with domain constraint $6 \le t \le 60$)*

### Desmos Calculator Input
```text
(t*cos(0.5236) - e^(0.03*|t|)*sin(0.3*t)*sin(0.5236) + 55, 42 + t*sin(0.5236) + e^(0.03*|t|)*sin(0.3*t)*cos(0.5236))
```

---

## 3. Mathematical Derivation & Method

Notice that the parametric system represents a spiral/oscillatory curve translated by $(X, 42)$ and rotated counter-clockwise by angle $\theta$. 

Translating by $(X, 42)$ and applying a clockwise rotation by angle $\theta$:
$$u = (x - X)\cos(\theta) + (y - 42)\sin(\theta) = t$$
$$v = -(x - X)\sin(\theta) + (y - 42)\cos(\theta) = e^{M |t|} \sin(0.3 t)$$

This decouples the parameter $t$ from $(x, y)$, allowing exact reconstruction and optimization with zero degrees of freedom.
