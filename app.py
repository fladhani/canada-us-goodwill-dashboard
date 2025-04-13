# app.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import streamlit as st

# Real-world data (simplified historical snapshots)
data = {
    "Year": list(range(2010, 2025)),
    "Favorable_Opinion_%": [80, 82, 81, 84, 85, 86, 85, 83, 82, 81, 83, 84, 85, 83, 82],
    "Trade_Volume_Bil_USD": [540, 560, 580, 600, 620, 640, 660, 655, 648, 650, 662, 700, 720, 740, 762],
    "Canadian_FDI_USD_Bil": [420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 630, 650, 670, 683, 700],
    "US_Tourism_Mil_Visitors": [11.5, 11.8, 12, 12.3, 12.6, 13, 13.3, 13.5, 13.8, 14, 14.2, 14.5, 14.8, 15, 14.6]
}
df = pd.DataFrame(data)

# Regression: Favorable Opinion → Trade Volume
X = sm.add_constant(df["Favorable_Opinion_%"])
y = df["Trade_Volume_Bil_USD"]
model = sm.OLS(y, X).fit()

# Granger Causality Test (Trade ↔ Favorability)
granger_data = df[["Trade_Volume_Bil_USD", "Favorable_Opinion_%"]]
granger_test = grangercausalitytests(granger_data, maxlag=2, verbose=False)

# STREAMLIT DASHBOARD
st.set_page_config(page_title="Canada-U.S. Goodwill Dashboard", layout="wide")
st.title("🇺🇸 American Goodwill and 🇨🇦 Canada-U.S. Prosperity Dashboard")

st.markdown("This dashboard explores how American public opinion toward Canada may influence bilateral prosperity indicators such as trade, investment, and tourism.")

# REGRESSION OUTPUT
st.header("📈 Regression Analysis")
st.text(model.summary())

# GRANGER CAUSALITY
st.header("🔁 Granger Causality Test (Does favorability drive trade?)")
for lag, result in granger_test.items():
    p_value = result[0]['ssr_chi2test'][1]
    st.write(f"Lag {lag}: p-value = {p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")

# TIME SERIES VISUALIZATION
st.header("📊 Time-Series Trends")
fig, ax = plt.subplots(3, 1, figsize=(10, 14))
sns.lineplot(x="Year", y="Favorable_Opinion_%", data=df, ax=ax[0]).set(title="Favorable Opinion of Canada (%)")
sns.lineplot(x="Year", y="Trade_Volume_Bil_USD", data=df, ax=ax[1]).set(title="U.S.-Canada Trade Volume (Billion USD)")
sns.lineplot(x="Year", y="Canadian_FDI_USD_Bil", data=df, ax=ax[2]).set(title="Canadian FDI in U.S. (Billion USD)")
plt.tight_layout()
st.pyplot(fig)

# CORRELATION MATRIX
st.header("🔎 Correlation Heatmap")
corr = df.drop(columns=["Year"]).corr()
fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
st.pyplot(fig2)

# FOOTER
st.markdown("---")
st.markdown("🧪 Data is illustrative and partially synthesized based on public sources (Pew, USTR, StatCan, Ipsos).")
