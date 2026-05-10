import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------
# Title
# --------------------------------

st.title("Retail Sales Forecasting System")

st.write("Enter monthly sales data to predict future sales.")

# --------------------------------
# User Input
# --------------------------------

n = st.number_input(
    "Enter number of months:",
    min_value=5,
    max_value=50,
    step=1
)

sales = []

st.subheader("Enter Sales Values")

for i in range(n):
    value = st.number_input(
        f"Month {i+1} Sales",
        min_value=0.0,
        step=1.0,
        key=i
    )
    sales.append(value)

# --------------------------------
# Forecast Button
# --------------------------------

if st.button("Generate Forecast"):

    # Create dataset
    dates = pd.date_range(
        start='2024-01-01',
        periods=n,
        freq='ME'
    )

    data = pd.DataFrame({
        'Month': dates,
        'Sales': sales
    })

    data.set_index('Month', inplace=True)

    # Display dataset
    st.subheader("Sales Dataset")
    st.write(data)

    # --------------------------------
    # Original Sales Graph
    # --------------------------------

    st.subheader("Sales Trend Graph")

    fig1, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(
        data.index,
        data['Sales'],
        marker='o'
    )

    ax1.set_title("Retail Sales Trend")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    ax1.grid(True)

    st.pyplot(fig1)

    # --------------------------------
    # ARIMA Forecasting
    # --------------------------------

    model = ARIMA(data['Sales'], order=(1,1,1))

    model_fit = model.fit()

    forecast_steps = 6

    forecast = model_fit.forecast(
        steps=forecast_steps
    )

    future_dates = pd.date_range(
        start=data.index[-1] + pd.DateOffset(months=1),
        periods=forecast_steps,
        freq='ME'
    )

    # --------------------------------
    # Forecast Values
    # --------------------------------

    st.subheader("Forecasted Sales")

    forecast_df = pd.DataFrame({
        'Month': future_dates,
        'Forecasted Sales': forecast
    })

    st.write(forecast_df)

    # --------------------------------
    # Forecast Graph
    # --------------------------------

    st.subheader("Forecast Graph")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(
        data.index,
        data['Sales'],
        marker='o',
        label='Original Sales'
    )

    ax2.plot(
        future_dates,
        forecast,
        marker='o',
        linestyle='dashed',
        label='Forecasted Sales'
    )

    ax2.set_title("Retail Sales Forecasting")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales")

    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)
    # --------------------------------
# Mathematical Framework
# --------------------------------

st.subheader("Mathematical Framework")

st.markdown("""
### Time Series Analysis

Time Series Analysis is used to analyze data collected over time
to identify trends and forecast future values.

### ARIMA Model

ARIMA stands for:

- **AR** → Auto Regression
- **I** → Integrated
- **MA** → Moving Average

The ARIMA model used in this project is:

ARIMA(1,1,1)

Where:

- p = 1 → autoregressive terms
- d = 1 → differencing
- q = 1 → moving average terms

### General ARIMA Equation

""")

st.latex(r'''
Y_t = c + \phi_1 Y_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t
''')

st.markdown("""

Where:

- \(Y_t\) = current value
- \(Y_{t-1}\) = previous value
- \(\phi\) = autoregressive coefficient
- \(\theta\) = moving average coefficient
- \(\epsilon\) = random error

### Forecasting Purpose

This model helps businesses:

- predict future sales
- manage inventory
- improve planning
- reduce financial risks

""")
# Custom Page Configuration

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide"
)
# --------------------------------
# Custom CSS Styling
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1 {
    color: #00FFFF;
    text-align: center;
}

h2, h3 {
    color: #FFD700;
}

.stButton>button {
    background-color: #00BFFF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stButton>button:hover {
    background-color: #1E90FF;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    color: white;
}

</style>
""", unsafe_allow_html=True)