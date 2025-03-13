
# Yield Curve Modeling: Multi-Curve and Greeks


https://github.com/user-attachments/assets/5ae5ea7c-2d0c-4ca1-9898-f476f1118835


This repository implements a yield curve generation model using interest rate swaps, Greek (delta) sensitivity analysis, and a multi-curve framework. The project uses numerical methods to calibrate the yield curve from market swap quotes and applies a bump-and-revalue approach to compute sensitivities. 

Repository URL: [https://github.com/shubh123a3/Yield-Curve-Modeling_Multi_Curve_and_Greek/tree/master](https://github.com/shubh123a3/Yield-Curve-Modeling_Multi_Curve_and_Greek/tree/master)

---

## Overview

The project consists of three main parts:
1. **Yield Curve Construction:** Calibrating a yield curve using swap instruments.
2. **Greeks (Delta) Calculation:** Measuring the sensitivity of swap prices to small changes in market quotes.
3. **Multi-Curve Framework:** Separating discounting and forecasting curves to improve pricing accuracy.

---

## 1. Yield Curve Construction

The yield curve is modeled by constructing a set of discount factors $P(0,T)$ based on continuously compounded yields. The discount factor is given by:

$$
P(0,T) = e^{-r(T) \cdot T},
$$

where $r(T)$ is the continuously compounded yield at maturity $T$. Market swap quotes are used to solve for the unknown yields (or spine points) at selected maturities via a numerical calibration process (e.g., Newton-Raphson).

### Newton-Raphson Calibration

We solve for the yield vector $\mathbf{r} = [r(T_1), r(T_2), \dots, r(T_N)]$ by setting up a system of equations derived from swap prices (which should be zero at par):

$$
f_i(\mathbf{r}) = 0,\quad i=1,2,\dots,N.
$$

The iterative update is given by:

$$
\mathbf{r}^{(n+1)} = \mathbf{r}^{(n)} - J^{-1} \cdot \mathbf{f}(\mathbf{r}^{(n)}),
$$

with the Jacobian matrix elements approximated by:

$$
J_{ij} \approx \frac{f_i(r_j + \epsilon) - f_i(r_j)}{\epsilon}.
$$

---

## 2. Swap Pricing

For a payer swap, the price is computed as:

$$
V_{\text{swap}} = \left[P(0,T_i) - P(0,T_m)\right] - K \cdot \sum_{j=1}^{n} \tau_j \, P(0,T_j),
$$

where:
- $T_i$ is the start date,
- $T_m$ is the maturity,
- $K$ is the fixed (strike) rate,
- $\tau_j$ is the accrual period for each payment date,
- $P(0,T_j)$ are the discount factors.

At calibration, we adjust $\mathbf{r}$ such that the modeled swap prices match the market quotes (typically zero for par swaps).

---

## 3. Greeks (Delta) Calculation

To assess the sensitivity of swap prices to changes in market quotes, we use a finite difference method (bump-and-revalue). If $S(K)$ is the swap price computed with a market quote $K$, a small change $dK$ results in a new price $S(K + dK)$. The delta is approximated by:

$$
\Delta \approx \frac{S(K + dK) - S(K)}{dK}.
$$

This computation is performed for each market quote, providing insight into how sensitive the swap is to changes in input rates.

---

## 4. Multi-Curve Framework

### Motivation

After the 2008 financial crisis, it became evident that a single yield curve was insufficient. Different curves are used for:
- **Discounting Cash Flows:** Typically derived from OIS rates.
- **Forecasting Forward Rates:** Based on LIBOR, EURIBOR, etc.

### Mathematical Formulation

#### Discount Curve
The discount factor from the OIS curve is:

$$
P_{\text{discount}}(0,T) = e^{-r_{\text{discount}}(T) \cdot T}.
$$

#### Forward Curve
The forward rate for a period $[T, T+\Delta]$ is derived as:

$$
F(0; T, T+\Delta) = \frac{1}{\Delta} \left( \frac{P_{\text{forward}}(0,T)}{P_{\text{forward}}(0,T+\Delta)} - 1 \right).
$$

#### Basis Spread
The difference between the LIBOR and OIS forward rates is modeled by the basis spread $S(T)$:

$$
F_{\text{LIBOR}}(0; T, T+\Delta) = F_{\text{OIS}}(0; T, T+\Delta) + S(T).
$$

#### Swap Pricing in Multi-Curve Environment
When pricing a swap using a multi-curve framework, the floating leg is projected using the forward curve and all cash flows are discounted using the discount curve:

$$
V_{\text{swap}} = \text{notional} \times \sum_{j=1}^{N} \tau_j \, P_{\text{discount}}(0, T_j) \left( F(0; T_{j-1}, T_j) - K \right).
$$

This approach ensures consistency with market practices by accurately reflecting funding costs and basis risks.

---

## Implementation

The project is implemented in Python using libraries such as NumPy, SciPy, Matplotlib, and Streamlit. Key functions include:

- **`IRSwap` and `IRSwapMultiCurve`:** Compute swap prices under single-curve and multi-curve settings.
- **`P0TModel` and `P0TModel_Greek`:** Compute discount factors.
- **`YieldCurve` and `MultivariateNewtonRaphson`:** Calibrate the yield curve via the Newton-Raphson method.
- **`BuildInstruments`:** Create swap instruments based on market quotes.
- **Visualization:** Plot yield curves, swap prices, and sensitivity (delta) analysis.

---

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/shubh123a3/Yield-Curve-Modeling_Multi_Curve_and_Greek.git
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application (using Streamlit for interactive visualization):
   ```
   streamlit run app.py
   ```

Follow the on-screen instructions to explore yield curve building, Greeks computation, and multi-curve analysis.

---

## Conclusion

This project implements a comprehensive yield curve modeling framework that includes:
- Calibration using market swap quotes,
- Greek (delta) sensitivity analysis through bump-and-revalue methods,
- Multi-curve techniques for accurate discounting and forecasting.

The use of LaTeX in the documentation clearly explains the theoretical and mathematical foundations behind each component, ensuring a deep understanding of modern interest rate modeling practices.

---



---

