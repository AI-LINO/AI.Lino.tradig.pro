import streamlit as st
import yfinance as yf
import numpy as np
from hmmlearn import hmm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino Pro", page_icon="🤖", layout="wide")

st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); 
     padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: #00ff88; font-size: 3em; font-family: Arial Black;'>🤖 AI.LINO PRO</h1>
    <h3 style='color: #ffffff;'>Máquina de Dinero — HMM Viterbi + RSI + MACD</h3>
    <p style='color: #aaaaaa;'>Señales de entrada y salida en tiempo real</p>
</div>
""", unsafe_allow_html=True)

def buscar_ticker(nombre):
    try:
        resultados = yf.Search(nombre, max_results=5)
        for q in resultados.quotes:
            if q.get("quoteType") in ["EQUITY", "ETF"]:
                ticker = q.get("symbol", "")
                nombre_real = q.get("longname") or q.get("shortname", ticker)
                exchange = q.get("exchange", "")
                if ticker.endswith(".MX"):
                    pais = "🇲🇽 México"
                elif any(ticker.endswith(x) for x in [".PA",".DE",".AS",".SW",".MC",".MI",".L"]):
                    pais = "🇪🇺 Europa"
                else:
                    pais = "🇺🇸 USA"
                return ticker, nombre_real, pais
    except:
        pass
    return None, None, None

def calcular_rsi(precios, periodo=14):
    delta = precios.diff()
    ganancia = delta.where(delta > 0, 0).rolling(periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs = ganancia / perdida
    return 100 - (100 / (1 + rs))

def calcular_macd(precios):
    ema12 = precios.ewm(span=12).mean()
    ema26 = precios.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

def calcular_score_final(estado_hmm, rsi, macd_val, signal_val, cambio):
    score = 50
    # HMM signal
    if estado_hmm == "ALCISTA":
        score += 25
    elif estado_hmm == "BAJISTA":
        score -= 25

    # RSI
    if rsi < 30:
        score += 20  # Sobreventa = oportunidad compra
    elif rsi > 70:
        score -= 20  # Sobrecompra = señal venta
    elif 40 < rsi < 60:
        score += 5

    # MACD
    if macd_val > signal_val:
        score += 15
    else:
        score -= 15

    # Momentum
    if cambio > 0:
        score += 5
    else:
        score -= 5

    return max(0, min(100, score))

class MaquinaDineroLino:
    def __init__(self):
        self.modelo = hmm.GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=2000, random_state=42, tol=1e-5
        )

    def analizar(self, ticker):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="2y")
            if df.empty or len(df) < 60:
                return None, "No hay suficientes datos"

            close = df["Close"].squeeze()
            retornos = np.log(close / close.shift(1)).dropna()
            volatilidad = retornos.rolling(5).std().dropna()
            momentum = close.pct_change(5).dropna()

            min_len = min(len(retornos), len(volatilidad), len(momentum))
            X = np.column_stack([
                retornos.values[-min_len:],
                volatilidad.values[-min_len:],
                momentum.values[-min_len:]
            ])

            self.modelo.fit(X)
            _, states = self.modelo.decode(X, algorithm="viterbi")

            means = self.modelo.means_[:, 0]
            bull_state = int(np.argmax(means))
            bear_state = int(np.argmin(means))
            estado_actual = int(states[-1])

            if estado_actual == bull_state:
                estado_hmm = "ALCISTA"
            elif estado_actual == bear_state:
                estado_hmm = "BAJISTA"
            else:
                estado_hmm = "LATERAL"

            # Indicadores técnicos
            rsi = calcular_rsi(close).iloc[-1]
            macd, signal = calcular_macd(close)
            macd_val = macd.iloc[-1]
            signal_val = signal.iloc[-1]

            precio_actual = float(close.iloc[-1])
            precio_ayer = float(close.iloc[-2])
            cambio = ((precio_actual - precio_ayer) / precio_ayer) * 100

            score = calcular_score_final(estado_hmm, rsi, macd_val, signal_val, cambio)

            if score >= 65:
                señal = "🟢 ENTRAR — COMPRAR"
                color = "#00ff88"
                accion = "COMPRAR"
            elif score <= 35:
                señal = "🔴 SALIR — VENDER"
                color = "#ff4444"
                accion = "VENDER"
            else:
                señal = "🟡 ESPERAR — NO OPERAR"
                color = "#ffcc00"
                accion = "ESPERAR"

            # Backtesting simple
            señales_bt = []
            for i in range(26, len(close)):
                r = calcular_rsi(close.iloc[:i+1]).iloc[-1]
                m, s = calcular_macd(close.iloc[:i+1])
                mv, sv = m.iloc[-1], s.iloc[-1]
                c = ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * 100
                sc = calcular_score_final(estado_hmm, r, mv, sv, c)
                señales_bt.append(sc)

            señales_series = pd.Series(señales_bt)
            compras = (señales_series >= 65).sum()
            ventas = (señales_series <= 35).sum()
            total = len(señales_series)

            return {
                "señal": señal, "score": score, "color": color,
                "accion": accion, "estado_hmm": estado_hmm,
                "precio": precio_actual, "cambio": cambio,
                "rsi": round(rsi, 1),
                "macd": round(macd_val, 4),
                "signal": round(signal_val, 4),
                "compras_bt": compras,
                "ventas_bt": ventas,
                "total_bt": total,
                "close": close,
                "datos": len(df)
            }, None

        except Exception as e:
            return None, str(e)

# --- INTERFAZ ---
busqueda = st.text_input("🔍 Escribe empresa o ticker:", 
                          placeholder="Ej: Tesla, Apple, FEMSA, Cemex...")

col1, col2 = st.columns([1, 3])
with col1:
    buscar_btn = st.button("🚀 ANALIZAR", use_container_width=True)
    auto_refresh = st.checkbox("🔄 Auto-refresh 15 min")

if auto_refresh:
    import time
    st.info("🔄 Actualizando cada 15 minutos...")
    time.sleep(900)
    st.rerun()

if buscar_btn and busqueda:
    with st.spinner("🔍 Buscando..."):
        ticker, nombre_real, pais = buscar_ticker(busqueda)

    if not ticker:
        st.error("❌ Empresa no encontrada")
    else:
        st.success(f"✅ {nombre_real} | {ticker} | {pais}")

        with st.spinner("🤖 AI.Lino analizando..."):
            maquina = MaquinaDineroLino()
            resultado, error = maquina.analizar(ticker)

        if error:
            st.error(f"Error: {error}")
        elif resultado:
            # Resultado principal
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f0c29, #302b63); 
                 padding: 30px; border-radius: 15px; text-align: center;
                 border: 3px solid {resultado["color"]}; margin: 15px 0;'>
                <h1 style='color: {resultado["color"]}; font-size: 3em;'>{resultado["señal"]}</h1>
                <h2 style='color: white;'>Score AI.LINO: {resultado["score"]}/100</h2>
                <h3 style='color: #aaaaaa;'>💰 Precio: ${resultado["precio"]:.2f} 
                <span style='color: {"#00ff88" if resultado["cambio"] > 0 else "#ff4444"}'>
                ({resultado["cambio"]:+.2f}%)</span></h3>
            </div>
            """, unsafe_allow_html=True)

            # Indicadores
            st.markdown("### 📊 Indicadores")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧠 HMM Viterbi", resultado["estado_hmm"])
            
            rsi_color = "inverse" if resultado["rsi"] > 70 else "normal"
            c2.metric("📈 RSI", resultado["rsi"],
                     "Sobrecompra" if resultado["rsi"] > 70 else 
                     "Sobreventa" if resultado["rsi"] < 30 else "Normal")
            
            macd_diff = round(resultado["macd"] - resultado["signal"], 4)
            c3.metric("📉 MACD", resultado["macd"], f"vs Signal: {macd_diff:+.4f}")
            c4.metric("📅 Datos", f"{resultado['datos']} días")

            # Backtesting
            st.markdown("### 🔬 Backtesting (2 años)")
            b1, b2, b3 = st.columns(3)
            b1.metric("🟢 Señales Compra", resultado["compras_bt"])
            b2.metric("🔴 Señales Venta", resultado["ventas_bt"])
            pct = round((resultado["compras_bt"] + resultado["ventas_bt"]) / resultado["total_bt"] * 100, 1)
            b3.metric("⚡ Actividad", f"{pct}%")

            # Gráfica
            st.markdown("### 📈 Precio Histórico")
            st.line_chart(resultado["close"])

            st.caption(f"🤖 AI.LINO Pro | HMM Viterbi + RSI + MACD | {ticker}")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#555;'>🤖 AI.Lino Pro © 2025 — Herramienta educativa. Opera con responsabilidad.</p>",
            unsafe_allow_html=True)