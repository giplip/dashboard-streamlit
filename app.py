import os
import subprocess

# Instalar alpha_vantage si no está presente
try:
    from alpha_vantage.timeseries import TimeSeries
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "alpha_vantage"])
    from alpha_vantage.timeseries import TimeSeries

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet

st.title("📈 Dashboard de Cotizaciones y Predicciones de Empresas")

companies = {
    "Palantir Technologies": "PLTR",
    "Symbotic": "SYM",
    "UiPath": "PATH",
    "CRISPR Therapeutics": "CRSP",
    "Exact Sciences": "EXAS",
    "Moderna": "MRNA",
    "Bloom Energy": "BE",
    "Solid Power": "SLDP",
    "Stem Inc.": "STEM",
    "Arista Networks": "ANET",
    "Lumen Technologies": "LUMN",
    "Digital Realty Trust": "DLR",
    "Rivian Automotive": "RIVN",
    "ChargePoint Holdings": "CHPT",
    "Mobileye Global": "MBLY"
}

selected_company = st.selectbox("Selecciona una empresa para visualizar:", list(companies.keys()))

# Reemplazamos yfinance con Alpha Vantage
API_KEY = "demo"  # 🔹 Para producción, usa tu propia API Key de Alpha Vantage
ts = TimeSeries(key=API_KEY, output_format="pandas")
data, meta_data = ts.get_daily(symbol=companies[selected_company], outputsize="compact")

st.subheader(f"Cotización histórica de {selected_company}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["4. close"], label="Precio de Cierre", color="blue")
ax.set_title(f"Cotización de {selected_company}")
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio de Cierre (USD)")
ax.legend()
st.pyplot(fig)

st.subheader(f"Predicción de Cotización para {selected_company}")

df = data.reset_index()
df = df[['date', '4. close']]
df.columns = ['ds', 'y']

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=180)
forecast = model.predict(future)

fig_forecast, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['ds'], df['y'], label="Histórico", color="blue")
ax.plot(forecast['ds'], forecast['yhat'], label="Predicción", color="red")
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color="pink", alpha=0.3)
ax.set_title(f"Predicción de Cotización para {selected_company}")
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio Estimado (USD)")
ax.legend()
st.pyplot(fig_forecast)

st.write("🔹 **Nota:** Las predicciones están basadas en modelos de series temporales y pueden no reflejar con precisión la evolución futura del mercado.")
