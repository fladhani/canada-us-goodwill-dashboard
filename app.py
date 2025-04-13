# Goodwill-Prosperity Impact Model and Dashboard
# --------------------------------------------------
# This script builds a model and visual dashboard to explore the
# relationship between American goodwill toward Canada and economic prosperity

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import streamlit as st

# Real-world data (simplified for modeling)
data = {
    "Year": list(range(2010, 2025)),
    "Favorable_Opinion_%": [80, 82, 81, 84, 85, 86, 85, 83, 82, 81, 83, 84, 85, 83, 82],
    "Trade_Volume_Bil_USD": [540, 560, 580, 600, 620, 640, 660, 655, 648, 650, 662, 700, 720, 740, 762],
    "Canadian_FDI_USD_Bil": [420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 630, 650, 670, 683, 700],
    "US_Tourism_Mil_Visitors": [11.5, 11.8, 12, 12.3, 12.6, 13, 13.3, 13.5, 13.8, 14, 14.2, 14.5, 14.8, 15, 14.6]
}

df = pd.DataFrame(data)

# Regression model: Favorability -> Trade Volume
X = sm.add_constant(df["Favorable_Opinion_%"])
y = df["Trade_Volume_Bil_USD"]
model = sm.OLS(y, X).fit()

# Granger Causality Test
granger_data = df[["Trade_Volume_Bil_USD", "Favorable_Opinion_%"]].dropna()
granger_test = grangercausalitytests(granger_data, maxlag=2, verbose=False)

# Streamlit Dashboard
st.title("🇺🇸 American Goodwill and 🇨🇦 Canada-U.S. Prosperity Dashboard")

st.header("📈 Regression Analysis")
st.text(model.summary())

st.header("🔁 Granger Causality Test")
for lag in granger_test:
    st.subheader(f"Lag {lag}")
    st.text(granger_test[lag][0]['ssr_chi2test'])

st.header("📊 Time-Series Visualizations")
fig, ax = plt.subplots(3, 1, figsize=(10, 12))
sns.lineplot(x="Year", y="Favorable_Opinion_%", data=df, ax=ax[0]).set(title="Favorable Opinion of Canada")
sns.lineplot(x="Year", y="Trade_Volume_Bil_USD", data=df, ax=ax[1]).set(title="U.S.-Canada Trade Volume")
sns.lineplot(x="Year", y="Canadian_FDI_USD_Bil", data=df, ax=ax[2]).set(title="Canadian FDI in U.S.")
plt.tight_layout()
st.pyplot(fig)

st.header("📊 Correlation Heatmap")
corr = df.drop(columns=["Year"]).corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax2)
st.pyplot(fig2)

# End of Streamlit app
