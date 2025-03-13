# Yield Curve Modeling, Multi-Curve Framework, and Greeks



https://github.com/user-attachments/assets/9df2f79d-77aa-4491-b00d-8443327f4e27


## Overview
This repository focuses on yield curve modeling, multi-curve frameworks, and Greeks calculation using swaps. The project implements mathematical and computational models to estimate and analyze yield curves, their sensitivity to interest rate movements, and the impact on derivatives pricing.

## Table of Contents
- [Introduction](#introduction)
- [Multi-Curve Framework](#multi-curve-framework)
- [Yield Curve Sensitivities and Greeks](#yield-curve-sensitivities-and-greeks)
- [Mathematical Formulation](#mathematical-formulation)
- [Implementation](#implementation)
- [Usage](#usage)
- [References](#references)

## Introduction
A **yield curve** represents the relationship between interest rates (or yields) and different maturities for bonds of similar credit quality. Traditional models use a **single-curve approach**, but post-2008 financial crisis, **multi-curve frameworks** have become essential due to discrepancies in discounting and forward rate curves.

### Why Multi-Curve Modeling?
- In the past, a single yield curve was used for discounting and forecasting future cash flows.
- Market disruptions led to the realization that different instruments require distinct yield curves.
- Multi-curve models differentiate between **discounting curves** (OIS) and **forward curves** for forecasting floating rates.

## Multi-Curve Framework
The multi-curve framework constructs separate curves for:
1. **OIS Discount Curve ($D(t,T)$)**: Used for discounting future cash flows.
2. **Forward Curve ($F(t,T)$)**: Used for pricing derivatives and forecasting forward rates.

The fundamental relationship between discount factors and forward rates is:
\[
F(t,T) = \frac{D(t,T)}{D(t,T+\delta)} - 1
\]
where:
- $D(t,T)$: Discount factor for time $T$
- $F(t,T)$: Forward rate for period $(T, T+\delta)$

### Curve Construction Methods
- **Bootstrapping**: Constructing discount and forward curves iteratively from market instruments like swaps, bonds, and FRAs.
- **Spline Interpolation**: Cubic splines or Nelson-Siegel models for smooth curve fitting.
- **Optimization Techniques**: Minimizing error between market-observed rates and model-implied rates.

## Yield Curve Sensitivities and Greeks
### Key Risk Measures:
1. **Delta ($\Delta$)**: Sensitivity of instrument price to a small change in the yield curve.
2. **Gamma ($\Gamma$)**: Second-order sensitivity measuring curvature effects.
3. **Vega ($\nu$)**: Sensitivity to changes in interest rate volatility.
4. **Theta ($\Theta$)**: Time decay impact on portfolio valuation.

Mathematically, the delta of a swap with respect to the yield curve is given by:
\[
\Delta = \frac{\partial V}{\partial r} = \sum_{i=1}^{n} \frac{\partial D(0,T_i)}{\partial r} C_i
\]
where:
- $V$: Swap value
- $r$: Interest rate
- $C_i$: Cash flows at time $T_i$

### Yield Curve Shifts (Shock Analysis)
Different shift methodologies include:
- **Parallel Shift**: Uniform movement in the entire yield curve.
- **Key Rate Shift**: Changes at specific maturities.
- **Twist & Butterfly Movements**: Adjustments affecting different parts of the curve differently.

## Mathematical Formulation
### Swap Pricing Formula
A plain vanilla interest rate swap's present value (PV) is given by:
\[
PV = \sum_{i=1}^{n} C_i D(0,T_i) - \sum_{j=1}^{m} F_j D(0,T_j)
\]
where:
- $C_i$: Fixed leg cash flows
- $F_j$: Floating leg payments
- $D(0,T_i)$: Discount factors

### Greeks Calculation
- **Delta ($\Delta$)**:
  \[
  \Delta = \frac{\partial V}{\partial r}
  \]
- **Gamma ($\Gamma$)**:
  \[
  \Gamma = \frac{\partial^2 V}{\partial r^2}
  \]
- **Vega ($\nu$)**:
  \[
  \nu = \frac{\partial V}{\partial \sigma}
  \]

## Implementation
The project is implemented in Python using:
- **Pandas & NumPy** for data manipulation
- **SciPy Optimization** for curve fitting
- **QuantLib** for financial modeling

## Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/shubh123a3/Yield-Curve-Modeling_Multi_Curve_and_Greek.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the model**:
   ```bash
   python main.py
   ```

## References
- John C. Hull, "Options, Futures, and Other Derivatives"
- Brigo & Mercurio, "Interest Rate Models – Theory and Practice"
- QuantLib Documentation

