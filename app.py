import os
os.system('pip install yfinance')

import streamlit as st
import yfinance as yf
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

start_date = datetime.now() - timedelta(days=5*365)
end_date = datetime.now()

symbol = companies[selected_company]
data = yf.download(symbol, start=start_date, end=end_date)

st.subheader(f"Cotización histórica de {selected_company}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data.index, data["Close"], label="Precio de Cierre", color="blue")
ax.set_title(f"Cotización de {selected_company} ({symbol})")
ax.set_xlabel("Fecha")
ax.set_ylabel("Precio de Cierre (USD)")
ax.legend()
st.pyplot(fig)

st.subheader(f"Predicción de Cotización para {selected_company}")

df = data.reset_index()
df = df[['Date', 'Close']]
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
