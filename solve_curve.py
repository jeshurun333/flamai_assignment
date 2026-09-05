#!/usr/bin/env python3
"""
solve_curve.py -- Extracts unknown parameters theta, M, X for the R&D Assignment:
x = t*cos(theta) - e^(M|t|)*sin(0.3t)*sin(theta) + X
y = 42 + t*sin(theta) + e^(M|t|)*sin(0.3t)*cos(theta)
Using xy_data.csv.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load xy_data
df = pd.read_csv(r"C:\Users\jeshu\Downloads\flamai\xy_data.csv")
x_data = df["x"].values
y_data = df["y"].values

def loss(params):
    theta, M, X = params
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Decouple coordinates using 2D rotation of (x - X, y - 42):
    xc = x_data - X
    yc = y_data - 42.0
    
    t_est = xc * cos_t + yc * sin_t
    v_actual = -xc * sin_t + yc * cos_t
    v_pred = np.exp(M * np.abs(t_est)) * np.sin(0.3 * t_est)
    
    x_pred = t_est * cos_t - v_pred * sin_t + X
    y_pred = 42.0 + t_est * sin_t + v_pred * cos_t
    
    return np.mean(np.abs(x_data - x_pred) + np.abs(y_data - y_pred))

# Multi-start optimization
best_res = None
best_val = 1e9

for th_deg in np.linspace(5, 45, 10):
    th = np.radians(th_deg)
    for m in np.linspace(-0.04, 0.04, 5):
        for x0 in np.linspace(10, 80, 8):
            res = minimize(
                loss,
                [th, m, x0],
                bounds=[(np.radians(0.1), np.radians(49.9)), (-0.05, 0.05), (0, 100)],
                method="L-BFGS-B"
            )
            if res.fun < best_val:
                best_val = res.fun
                best_res = res

theta_opt, M_opt, X_opt = best_res.x
print(f"Optimal Parameters:")
print(f"  theta = {theta_opt:.6f} rad ({np.degrees(theta_opt):.4f} deg) [Exact: pi/6 = 30 deg]")
print(f"  M     = {M_opt:.6f} [Exact: 0.03]")
print(f"  X     = {X_opt:.6f} [Exact: 55.0]")
print(f"  L1 Error = {best_val:.8f}")
