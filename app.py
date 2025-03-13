import numpy as np
import enum
from copy import deepcopy
from scipy.interpolate import splrep, splev, interp1d
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Yield Curve Building")


class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0


def IRSwap(CP, notional, K, t, Ti, Tm, n, P0T):
    # CP- payer or receiver
    # notional- notional amount
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # P0T - Zero coupon bond price function

    # Create time grid for payments
    ti_grid = np.linspace(Ti, Tm, int(n))
    tau = ti_grid[1] - ti_grid[0]

    # Filter time points if t is after some payment dates
    ti_grid = ti_grid[np.where(ti_grid > t)]

    temp = 0.0
    # Calculate discounted cash flows
    for (idx, ti) in enumerate(ti_grid):
        if idx > 0:  # Start from second payment
            temp = temp + tau * P0T(ti)

    # Calculate discount factors for start and end
    P_t_Ti = P0T(Ti)
    P_t_Tm = P0T(Tm)

    # Calculate swap value based on option type
    if CP == OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    elif CP == OptionTypeSwap.RECEIVER:
        swap = K * temp - (P_t_Ti - P_t_Tm)

    return swap * notional


def P0TModel(t, ti, ri, method):
    rInterp = method(ti, ri)
    r = rInterp(t)
    return np.exp(-r * t)


def P0TModel_Greek(t, ti, ri, method):
    rInterp = method(ti, ri)
    if t >= ti[-1]:
        r = ri[-1]
    elif t <= ti[0]:
        r = ri[0]
    else:
        r = rInterp(t)
    return np.exp(-r * t)


def IRSwapMultiCurve(CP, notional, K, t, Ti, Tm, n, P0T, P0TFrd):
    # Multi-curve version using separate discount and forward curves
    ti_grid = np.linspace(Ti, Tm, int(n))
    tau = ti_grid[1] - ti_grid[0]
    swap = 0.0

    for (idx, ti) in enumerate(ti_grid):
        if idx > 0:
            # Calculate forward rate from forward curve
            L_frwd = 1.0 / tau * (P0TFrd(ti_grid[idx - 1]) - P0TFrd(ti_grid[idx])) / P0TFrd(ti_grid[idx])
            # Discount with discount curve
            swap = swap + tau * P0T(ti_grid[idx]) * (L_frwd - K)

    return swap * notional


def YieldCurve(instruments, maturities, r0, method, tol):
    r0 = deepcopy(r0)
    ri = MultivariateNewtonRaphson(r0, maturities, instruments, method, tol=tol)
    return ri


def MultivariateNewtonRaphson(ri, ti, instruments, method, tol):
    err = 10e10
    idx = 0
    while err > tol:
        idx = idx + 1
        values = EvaluateInstruments(ti, ri, instruments, method)
        J = Jacobian(ti, ri, instruments, method)
        J_inv = np.linalg.inv(J)
        err = - np.dot(J_inv, values)
        ri[0:] = ri[0:] + err
        err = np.linalg.norm(err)
        print('index in the loop is', idx, ' Error is ', err)
    return ri


def Jacobian(ti, ri, instruments, method):
    eps = 1e-05
    swap_num = len(ti)
    J = np.zeros([swap_num, swap_num])
    val = EvaluateInstruments(ti, ri, instruments, method)
    ri_up = deepcopy(ri)

    for j in range(0, len(ri)):
        ri_up[j] = ri_up[j] + eps
        val_up = EvaluateInstruments(ti, ri_up, instruments, method)
        ri_up[j] = ri[j]
        dv = (val_up - val) / eps
        J[:, j] = dv[:]
    return J


def EvaluateInstruments(ti, ri, instruments, method):
    P0Ttemp = lambda t: P0TModel(t, ti, ri, method)
    val = np.zeros(len(instruments))
    for i in range(0, len(instruments)):
        val[i] = instruments[i](P0Ttemp)
    return val


def linear_interpolation(ti, ri):
    interpolator = lambda t: np.interp(t, ti, ri)
    return interpolator


def quadratic_interpolation(ti, ri):
    interpolator = interp1d(ti, ri, kind='quadratic', fill_value="extrapolate")
    return interpolator


def cubic_interpolation(ti, ri):
    interpolator = interp1d(ti, ri, kind='cubic', fill_value="extrapolate")
    return interpolator


def BuildInstruments(K, mat):
    swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[0], 0.0, 0.0, mat[0], 4 * mat[0], P0T)
    swap2 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[1], 0.0, 0.0, mat[1], 4 * mat[1], P0T)
    swap3 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[2], 0.0, 0.0, mat[2], 4 * mat[2], P0T)
    swap4 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[3], 0.0, 0.0, mat[3], 4 * mat[3], P0T)
    swap5 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[4], 0.0, 0.0, mat[4], 4 * mat[4], P0T)
    swap6 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[5], 0.0, 0.0, mat[5], 4 * mat[5], P0T)
    swap7 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[6], 0.0, 0.0, mat[6], 4 * mat[6], P0T)
    swap8 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[7], 0.0, 0.0, mat[7], 4 * mat[7], P0T)
    instruments = [swap1, swap2, swap3, swap4, swap5, swap6, swap7, swap8]
    return instruments


def main():
    st.sidebar.title("Yield Curve Building")
    st.sidebar.write("This is a simple implementation of yield curve building using Newton Raphson method.")

    Button = st.sidebar.button("Build Yield Curve")
    Greek_Button = st.sidebar.button("Build Yield Curve with Greeks")
    Multi_Curve_Button = st.sidebar.button("Build Multi-Curve")
    tol = 1.0e-15

    # Initial guess for the spine points
    r0 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

    K = np.array([0.04 / 100.0, 0.16 / 100.0, 0.31 / 100.0, 0.81 / 100.0, 1.28 / 100.0, 1.62 / 100.0, 2.22 / 100.0,
                  2.30 / 100.0])
    mat = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    t_line = np.linspace(mat.min(), mat.max(), 300)
    K_cont = np.interp(t_line, mat, K)
    fig, ax = plt.subplots()
    ax.plot(t_line, K_cont)
    ax.scatter(mat, K, c='r')
    ax.set_title("Original spine point(bonds) ")
    ax.set_xlabel("maturity ")
    ax.set_ylabel("yield")
    st.pyplot(fig)

    # UI components
    method_dict = {"Linear Interpolation": linear_interpolation,
                   "Cubic Interpolation": cubic_interpolation,
                   "Quadratic Interpolation": quadratic_interpolation}
    method_name = st.sidebar.selectbox("Select Interpolation Method", list(method_dict.keys()))
    method = method_dict[method_name]  # Get the actual function from the dictionary

    Optiontype = st.sidebar.radio("Option Type", [OptionTypeSwap.PAYER, OptionTypeSwap.RECEIVER])
    notional = st.sidebar.number_input("Notional", min_value=1.0)
    K_one = st.sidebar.number_input("Strike", min_value=0.0, value=0.03)
    t = st.sidebar.number_input("Today's Date", min_value=0.0, value=0.0)
    Ti = st.sidebar.number_input("Beginning of the Swap", min_value=0.0, value=0.0)
    Tm = st.sidebar.number_input("End of Swap", min_value=0.0, max_value=30.0, step=1.0, value=4.0)
    n = st.sidebar.number_input("Number of Dates Payments between Ti and Tm", min_value=1.0, value=6.0)

    if Button:
        tol = 1.0e-15
        swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[0], 0.0, 0.0, mat[0], 4 * mat[0], P0T)
        swap2 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[1], 0.0, 0.0, mat[1], 4 * mat[1], P0T)
        swap3 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[2], 0.0, 0.0, mat[2], 4 * mat[2], P0T)
        swap4 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[3], 0.0, 0.0, mat[3], 4 * mat[3], P0T)
        swap5 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[4], 0.0, 0.0, mat[4], 4 * mat[4], P0T)
        swap6 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[5], 0.0, 0.0, mat[5], 4 * mat[5], P0T)
        swap7 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[6], 0.0, 0.0, mat[6], 4 * mat[6], P0T)
        swap8 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[7], 0.0, 0.0, mat[7], 4 * mat[7], P0T)
        instruments = [swap1, swap2, swap3, swap4, swap5, swap6, swap7, swap8]
        st.subheader("Yield Curve Building")
        method = linear_interpolation
        ri = YieldCurve(instruments, mat, r0, method, tol)


        st.subheader("Spine points are")
        # adding amturity to spine point
        for i in range(len(mat)):
            st.write(f"Maturity: {mat[i]} years, Spine Point: {ri[i]:.6f}")



        # building the zcb -curve/yeild
        P0T_Initial = lambda t: P0TModel(t, mat, r0, method)
        P0T = lambda t: P0TModel(t, mat, ri, method)

        # price back the swap
        swapsModel = np.zeros(len(instruments))
        swapIntial = np.zeros(len(instruments))
        for i in range(0, len(instruments)):
            swapIntial[i] = instruments[i](P0T_Initial)
            swapsModel[i] = instruments[i](P0T)

        st.subheader("Swap Prices")
        st.success(f"Prices for  Swaps (initial) = {swapIntial}")
        st.subheader("Swap Par Prices")
        st.success(f"Prices for Par Swaps = {swapsModel}")

    if Greek_Button:
        instruments = BuildInstruments(K, mat)
        ri = YieldCurve(instruments, mat, r0, method, tol)
        P0T = lambda t: P0TModel_Greek(t, mat, ri, method)

        SwapLambda = lambda P0T: IRSwap(Optiontype, notional, K_one, t, Ti, Tm, n, P0T)
        swap = SwapLambda(P0T)
        st.subheader('Swap price')
        st.success(f"Swap price= {swap:.6f}")

        # Plot swap price
        t_grid = np.linspace(Ti, Tm, 100)
        swap_grid = np.zeros(len(t_grid))
        for i in range(0, len(t_grid)):
            # Call IRSwap directly with different t values
            swap_grid[i] = IRSwap(Optiontype, notional, K_one, t_grid[i], Ti, Tm, n, P0T)
        # Plotly plot
        fig = px.line(x=t_grid, y=swap_grid, title='Swap Price')
        st.plotly_chart(fig)

        dK = 0.0001
        delta = np.zeros(len(K))
        K_new = np.copy(K)
        for i in range(0, len(K)):
            K_new[i] = K_new[i] + dK  # Bump the i-th element
            instruments = BuildInstruments(K_new, mat)
            ri = YieldCurve(instruments, mat, r0, method, tol)
            P0T_new = lambda t: P0TModel_Greek(t, mat, ri, method)
            swap_shock = SwapLambda(P0T_new)
            delta[i] = (swap_shock - swap) / dK
            K_new[i] = K_new[i] - dK  # Reset the bumped value

        for i in range(len(mat)):
            st.success(f"Maturity: {mat[i]} years, Delta: {delta[i]:.6f}")

        st.write(
            'Here are the delta values for the given maturities that can be used for hedging the risk of the swap price by entering into other swaps with different maturities and strikes.')

    if Multi_Curve_Button:
        st.header('Multi Curve Building')
        st.write('This is a simple implementation of multi curve building using Newton Raphson method.')

        # Convergence tolerance
        tol = 1.0e-15
        # Initial guess for the spine points
        r0 = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

        # Construct swaps that are used for building of the yield curve
        K = np.array([0.04 / 100.0, 0.16 / 100.0, 0.31 / 100.0, 0.81 / 100.0, 1.28 / 100.0, 1.62 / 100.0, 2.22 / 100.0,
                      2.30 / 100.0])
        mat = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

        # Construct instruments for discount curve
        swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[0], 0.0, 0.0, mat[0], 4 * mat[0], P0T)
        swap2 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[1], 0.0, 0.0, mat[1], 4 * mat[1], P0T)
        swap3 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[2], 0.0, 0.0, mat[2], 4 * mat[2], P0T)
        swap4 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[3], 0.0, 0.0, mat[3], 4 * mat[3], P0T)
        swap5 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[4], 0.0, 0.0, mat[4], 4 * mat[4], P0T)
        swap6 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[5], 0.0, 0.0, mat[5], 4 * mat[5], P0T)
        swap7 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[6], 0.0, 0.0, mat[6], 4 * mat[6], P0T)
        swap8 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, K[7], 0.0, 0.0, mat[7], 4 * mat[7], P0T)
        instruments = [swap1, swap2, swap3, swap4, swap5, swap6, swap7, swap8]

        # Determine optimal spine points for discount curve
        ri = YieldCurve(instruments, mat, r0, method, tol)
        st.subheader('Discount Curve Spine points')
        st.success(f"Spine points: {ri}")

        # Build a ZCB-curve/yield curve from the spine points
        P0T_Initial = lambda t: P0TModel(t, mat, r0, method)
        P0T = lambda t: P0TModel(t, mat, ri, method)

        # Price back the swaps
        swapsModel = np.zeros(len(instruments))
        swapsInitial = np.zeros(len(instruments))
        for i in range(0, len(instruments)):
            swapsModel[i] = instruments[i](P0T)
            swapsInitial[i] = instruments[i](P0T_Initial)

        st.subheader("Prices for Swaps (initial)")
        st.success(f"Prices for Swaps (initial) = {swapsInitial}")
        st.subheader("Prices for par Swaps")
        st.success(f"Prices for Par Swaps = {swapsModel}")

        # Multi Curve extension - simple sanity check
        P0TFrd = deepcopy(P0T)  # Initial test with same curve
        Ktest = 0.002  # Small strike for better visualization of the difference
        swap1 = lambda P0T: IRSwap(OptionTypeSwap.PAYER, 1, Ktest, 0.0, 0.0, mat[0], 4 * mat[0], P0T)
        swap1MC = lambda P0T: IRSwapMultiCurve(OptionTypeSwap.PAYER, 1, Ktest, 0.0, 0.0, mat[0], 4 * mat[0], P0T,
                                               P0TFrd)

        sanity_result1 = swap1(P0T)
        sanity_result2 = swap1MC(P0T)
        st.subheader("Sanity Check")
        st.success(f'Sanity check: swap1 = {sanity_result1:.8f}, swap1MC = {sanity_result2:.8f}')

        if abs(sanity_result1 - sanity_result2) < 1e-10:
            st.success("✓ Sanity check passed! The functions match as expected.")
        else:
            st.error(f"✗ Sanity check failed! Difference: {abs(sanity_result1 - sanity_result2):.8f}")

        # Forward curve settings
        r0Frwd = np.array([0.01, 0.01, 0.01, 0.01])
        KFrwd = np.array([0.09 / 100.0, 0.26 / 100.0, 0.37 / 100.0, 1.91 / 100.0])
        matFrwd = np.array([1.0, 2.0, 3.0, 5.0])

        # At this point we already know P(0,T) for the discount curve
        P0TDiscount = lambda t: P0TModel(t, mat, ri, method)

        # Build forward curve instruments
        swap1Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER, 1, KFrwd[0], 0.0, 0.0, matFrwd[0],
                                                     4 * matFrwd[0], P0TDiscount, P0TFrwd)
        swap2Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER, 1, KFrwd[1], 0.0, 0.0, matFrwd[1],
                                                     4 * matFrwd[1], P0TDiscount, P0TFrwd)
        swap3Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER, 1, KFrwd[2], 0.0, 0.0, matFrwd[2],
                                                     4 * matFrwd[2], P0TDiscount, P0TFrwd)
        swap4Frwd = lambda P0TFrwd: IRSwapMultiCurve(OptionTypeSwap.PAYER, 1, KFrwd[3], 0.0, 0.0, matFrwd[3],
                                                     4 * matFrwd[3], P0TDiscount, P0TFrwd)

        instrumentsFrwd = [swap1Frwd, swap2Frwd, swap3Frwd, swap4Frwd]

        # Determine optimal spine points for the forward curve
        riFrwd = YieldCurve(instrumentsFrwd, matFrwd, r0Frwd, method, tol)
        st.subheader('Forward Curve Spine points')
        st.success(f"Forward Spine points: {riFrwd}")

        # Build a ZCB-curve/yield curve from the spine points
        P0TFrwd_Initial = lambda t: P0TModel(t, matFrwd, r0Frwd, method)
        P0TFrwd = lambda t: P0TModel(t, matFrwd, riFrwd, method)

        # Price back the swaps
        swapsModelFrwd = np.zeros(len(instrumentsFrwd))
        swapsInitialFrwd = np.zeros(len(instrumentsFrwd))

        for i in range(0, len(instrumentsFrwd)):
            swapsModelFrwd[i] = instrumentsFrwd[i](P0TFrwd)
            swapsInitialFrwd[i] = instrumentsFrwd[i](P0TFrwd_Initial)

        st.subheader("Forward Curve Swap Prices")
        st.success(f"Initial Prices = {swapsInitialFrwd}")
        st.success(f"Par Prices = {swapsModelFrwd}")

        st.subheader('Price of one forward swap')
        forward_swap_price = swap1Frwd(P0TFrwd)
        st.success(f"Forward swap price: {forward_swap_price:.8f}")

        # Plot both curves
        fig, ax = plt.subplots(figsize=(10, 6))
        t = np.linspace(0, 10, 100)

        discount_values = [P0TDiscount(ti) for ti in t]
        forward_values = [P0TFrwd(ti) for ti in t]

        ax.plot(t, discount_values, '--r', linewidth=2)
        ax.plot(t, forward_values, '-b', linewidth=2)
        ax.legend(['Discount Curve', 'Forward Curve'], fontsize=12)
        ax.set_title('Discount vs Forward Curve', fontsize=14)
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('ZCB Price P(0,T)', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Calculate and display forward rates
        st.subheader("Forward Rates")
        tau = 0.25  # 3-month forward rate
        forward_times = np.arange(0.25, 5.0, 0.25)
        forward_rates = []

        for t in forward_times:
            # Calculate forward rate using the formula L(t,T) = (P(0,t)/P(0,t+τ) - 1)/τ
            fwd_rate = (P0TFrwd(t) / P0TFrwd(t + tau) - 1) / tau
            forward_rates.append(fwd_rate * 100)  # Convert to percentage

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(forward_times, forward_rates, '-g', linewidth=2)
        ax.set_title('3-Month Forward Rates', fontsize=14)
        ax.set_xlabel('Start Time (years)', fontsize=12)
        ax.set_ylabel('Forward Rate (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)


if __name__ == "__main__":
    main()