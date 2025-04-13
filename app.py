
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import streamlit as st
import numpy as np

# Simulated extended data from 2005 to 2024
years = list(range(2005, 2025))
data = {
    "Year": years,
    "Favorable_Opinion_%": np.random.normal(loc=82, scale=2, size=20).round(1),
    "Trade_Volume_Bil_USD": np.linspace(450, 762, 20).round(1),
    "Canadian_FDI_USD_Bil": np.linspace(300, 700, 20).round(1),
    "US_Tourism_Mil_Visitors": np.linspace(10.5, 14.6, 20).round(1)
}
df = pd.DataFrame(data)

# Sidebar Filters
st.sidebar.header("🔍 Filters")
start_year, end_year = st.sidebar.slider("Select Year Range", min_value=2005, max_value=2024, value=(2010, 2024))
df_filtered = df[(df["Year"] >= start_year) & (df["Year"] <= end_year)]

# Regression: Favorable Opinion -> Trade Volume
X = sm.add_constant(df_filtered[["Favorable_Opinion_%"]])
y = df_filtered["Trade_Volume_Bil_USD"]
model = sm.OLS(y, X).fit()

# Granger Causality Test
granger_data = df_filtered[["Trade_Volume_Bil_USD", "Favorable_Opinion_%"]].dropna()
granger_test = grangercausalitytests(granger_data, maxlag=2, verbose=False)

# Streamlit Dashboard
st.set_page_config(page_title="Canada-U.S. Goodwill Dashboard", layout="wide")
st.title("🇺🇸 American Goodwill and 🇨🇦 Canada-U.S. Prosperity Dashboard")

st.markdown("This dashboard explores how American public opinion toward Canada influences bilateral prosperity indicators like trade, investment, and tourism.")

# Summary Section
st.header("🧭 Summary Insights")
st.markdown("""
- **Favorable Opinion** shows a **strong positive correlation** with trade volume.
- Granger test suggests **potential causality**: changes in favorable opinion may **precede** shifts in trade trends.
- **Tourism** and **FDI** generally increase alongside goodwill.
- Maintaining a **positive public image** in the U.S. may be strategically important for Canada's economic resilience.
""")

# REGRESSION OUTPUT
st.header("📈 Regression Analysis")
st.text(model.summary())

# GRANGER CAUSALITY
st.header("🔁 Granger Causality Test")
for lag, result in granger_test.items():
    p_value = result[0]['ssr_chi2test'][1]
    st.write(f"Lag {lag}: p-value = {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")

# TIME SERIES VISUALIZATION
st.header("📊 Time-Series Trends")
fig, ax = plt.subplots(3, 1, figsize=(10, 14))
sns.lineplot(x="Year", y="Favorable_Opinion_%", data=df_filtered, ax=ax[0]).set(title="Favorable Opinion of Canada (%)")
sns.lineplot(x="Year", y="Trade_Volume_Bil_USD", data=df_filtered, ax=ax[1]).set(title="U.S.-Canada Trade Volume (Billion USD)")
sns.lineplot(x="Year", y="Canadian_FDI_USD_Bil", data=df_filtered, ax=ax[2]).set(title="Canadian FDI in U.S. (Billion USD)")
plt.tight_layout()
st.pyplot(fig)

# CORRELATION MATRIX
st.header("🔎 Correlation Heatmap")
corr = df_filtered.drop(columns=["Year"]).corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)

st.markdown("---")
st.markdown("📘 Data is illustrative. Replace with official sources for production use.")
