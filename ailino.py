import streamlit as st
import yfinance as yf
import numpy as np
from hmmlearn import hmm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# ─────────────────────────────────────────────
# FUNCIONES CORE
# ─────────────────────────────────────────────

def buscar_sugerencias(texto):
    """Devuelve lista de sugerencias usando yf.Search"""
    try:
        resultados = yf.Search(texto, max_results=8)
        sugerencias = []
        for q in resultados.quotes:
            if q.get("quoteType") in ["EQUITY", "ETF", "CRYPTOCURRENCY"]:
                ticker = q.get("symbol", "")
                nombre = q.get("longname") or q.get("shortname", ticker)
                tipo = q.get("quoteType", "")
                if ticker.endswith(".MX"):
                    pais = "🇲🇽"
                elif any(ticker.endswith(x) for x in [".PA", ".DE", ".AS", ".SW", ".MC", ".MI", ".L"]):
                    pais = "🇪🇺"
                elif tipo == "CRYPTOCURRENCY":
                    pais = "🪙"
                else:
                    pais = "🇺🇸"
                sugerencias.append({
                    "label": f"{pais} {nombre} ({ticker})",
                    "ticker": ticker,
                    "nombre": nombre,
                    "pais": pais
                })
        return sugerencias
    except:
        return []

def obtener_precio_realtime(ticker):
    """Precio en tiempo real usando fast_info — mucho más rápido que .info"""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        precio = fi.last_price
        precio_prev = fi.previous_close
        if precio and precio_prev:
            cambio_pct = ((precio - precio_prev) / precio_prev) * 100
            return precio, cambio_pct
    except:
        pass
    return None, None

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
    histograma = macd - signal
    return macd, signal, histograma

def calcular_score_final(estado_hmm, rsi, macd_val, signal_val, cambio):
    score = 50
    if estado_hmm == "ALCISTA":
        score += 25
    elif estado_hmm == "BAJISTA":
        score -= 25
    if rsi < 30:
        score += 20
    elif rsi > 70:
        score -= 20
    elif 40 < rsi < 60:
        score += 5
    if macd_val > signal_val:
        score += 15
    else:
        score -= 15
    if cambio > 0:
        score += 5
    else:
        score -= 5
    return max(0, min(100, score))

def grafica_trading_profesional(df_intraday, ticker, nombre):
    """
    Gráfica de velas japonesas intradía con:
    - Velas japonesas (OHLC)
    - Volumen con colores
    - EMA 9 y EMA 21
    - RSI
    - MACD con histograma
    """
    if df_intraday.empty or len(df_intraday) < 10:
        return None

    close = df_intraday['Close'].squeeze()
    ema9  = close.ewm(span=9).mean()
    ema21 = close.ewm(span=21).mean()
    rsi   = calcular_rsi(close, 14)
    macd, signal, histograma = calcular_macd(close)

    # Colores de velas
    colores_velas = ['#00ff88' if c >= o else '#ff4444'
                     for c, o in zip(df_intraday['Close'], df_intraday['Open'])]
    colores_vol   = ['#00ff88' if c >= o else '#ff4444'
                     for c, o in zip(df_intraday['Close'], df_intraday['Open'])]
    colores_hist  = ['#00ff88' if v >= 0 else '#ff4444' for v in histograma]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.15, 0.18, 0.17],
        subplot_titles=(
            f"📈 {nombre} ({ticker}) — Velas 1D",
            "📊 Volumen",
            "⚡ RSI (14)",
            "📉 MACD (12,26,9)"
        )
    )

    # ── PANEL 1: Velas + EMAs ──
    fig.add_trace(go.Candlestick(
        x=df_intraday.index,
        open=df_intraday['Open'],
        high=df_intraday['High'],
        low=df_intraday['Low'],
        close=df_intraday['Close'],
        name="Precio",
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88',
        decreasing_fillcolor='#ff4444',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_intraday.index, y=ema9,
        name="EMA 9", line=dict(color='#FFD700', width=1.5, dash='solid')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_intraday.index, y=ema21,
        name="EMA 21", line=dict(color='#00BFFF', width=1.5, dash='solid')
    ), row=1, col=1)

    # ── PANEL 2: Volumen ──
    fig.add_trace(go.Bar(
        x=df_intraday.index,
        y=df_intraday['Volume'],
        name="Volumen",
        marker_color=colores_vol,
        opacity=0.7
    ), row=2, col=1)

    # ── PANEL 3: RSI ──
    fig.add_trace(go.Scatter(
        x=df_intraday.index, y=rsi,
        name="RSI", line=dict(color='#DA70D6', width=2)
    ), row=3, col=1)

    # Zonas RSI
    fig.add_hrect(y0=70, y1=100, fillcolor="#ff4444", opacity=0.1,
                  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="#00ff88", opacity=0.1,
                  line_width=0, row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#ff4444",
                  line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00ff88",
                  line_width=1, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#888888",
                  line_width=1, row=3, col=1)

    # ── PANEL 4: MACD ──
    fig.add_trace(go.Bar(
        x=df_intraday.index, y=histograma,
        name="Histograma", marker_color=colores_hist, opacity=0.7
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df_intraday.index, y=macd,
        name="MACD", line=dict(color='#00BFFF', width=1.5)
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df_intraday.index, y=signal,
        name="Signal", line=dict(color='#FF8C00', width=1.5)
    ), row=4, col=1)

    fig.add_hline(y=0, line_dash="dot", line_color="#888888",
                  line_width=1, row=4, col=1)

    # ── ESTILO OSCURO PROFESIONAL ──
    fig.update_layout(
        height=750,
        paper_bgcolor='#0f0c29',
        plot_bgcolor='#0f0c29',
        font=dict(color='#ffffff', size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor='rgba(0,0,0,0.3)',
            font=dict(size=10)
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
    )

    # Grids y ejes
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=True, gridcolor='#1e1e3a',
            zeroline=False, row=i, col=1,
            showspikes=True, spikecolor="#00ff88",
            spikethickness=1, spikemode="across"
        )
        fig.update_yaxes(
            showgrid=True, gridcolor='#1e1e3a',
            zeroline=False, row=i, col=1,
            tickfont=dict(size=9)
        )

    # RSI rango fijo
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    return fig


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
            retornos    = np.log(close / close.shift(1)).dropna()
            volatilidad = retornos.rolling(5).std().dropna()
            momentum    = close.pct_change(5).dropna()

            min_len = min(len(retornos), len(volatilidad), len(momentum))
            X = np.column_stack([
                retornos.values[-min_len:],
                volatilidad.values[-min_len:],
                momentum.values[-min_len:]
            ])

            self.modelo.fit(X)
            _, states = self.modelo.decode(X, algorithm="viterbi")

            means       = self.modelo.means_[:, 0]
            bull_state  = int(np.argmax(means))
            bear_state  = int(np.argmin(means))
            estado_actual = int(states[-1])

            if estado_actual == bull_state:
                estado_hmm = "ALCISTA"
            elif estado_actual == bear_state:
                estado_hmm = "BAJISTA"
            else:
                estado_hmm = "LATERAL"

            rsi_serie       = calcular_rsi(close)
            rsi             = rsi_serie.iloc[-1]
            macd, signal, _ = calcular_macd(close)
            macd_val        = macd.iloc[-1]
            signal_val      = signal.iloc[-1]

            # Precio en tiempo real (fast_info — sin latencia)
            precio_rt, cambio_rt = obtener_precio_realtime(ticker)
            if precio_rt:
                precio_actual = precio_rt
                cambio        = cambio_rt
            else:
                precio_actual = float(close.iloc[-1])
                precio_ayer   = float(close.iloc[-2])
                cambio        = ((precio_actual - precio_ayer) / precio_ayer) * 100

            score = calcular_score_final(estado_hmm, rsi, macd_val, signal_val, cambio)

            if score >= 65:
                señal  = "🟢 ENTRAR — COMPRAR"
                color  = "#00ff88"
                accion = "COMPRAR"
            elif score <= 35:
                señal  = "🔴 SALIR — VENDER"
                color  = "#ff4444"
                accion = "VENDER"
            else:
                señal  = "🟡 ESPERAR — NO OPERAR"
                color  = "#ffcc00"
                accion = "ESPERAR"

            # Backtesting simple
            señales_bt = []
            for i in range(26, len(close)):
                r        = calcular_rsi(close.iloc[:i+1]).iloc[-1]
                m, s, _  = calcular_macd(close.iloc[:i+1])
                mv, sv   = m.iloc[-1], s.iloc[-1]
                c        = ((close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]) * 100
                sc       = calcular_score_final(estado_hmm, r, mv, sv, c)
                señales_bt.append(sc)

            señales_series = pd.Series(señales_bt)
            compras = (señales_series >= 65).sum()
            ventas  = (señales_series <= 35).sum()
            total   = len(señales_series)

            # Datos intradía para gráfica (1 año, velas diarias)
            df_chart = t.history(period="1y", interval="1d")

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
                "df_chart": df_chart,
                "datos": len(df)
            }, None

        except Exception as e:
            return None, str(e)


# ─────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────

# Session state para ticker seleccionado
if "ticker_sel"  not in st.session_state:
    st.session_state.ticker_sel  = ""
if "nombre_sel"  not in st.session_state:
    st.session_state.nombre_sel  = ""
if "pais_sel"    not in st.session_state:
    st.session_state.pais_sel    = ""
if "sugerencias" not in st.session_state:
    st.session_state.sugerencias = []

# Búsqueda con autocompletado
col_search, col_btn = st.columns([4, 1])

with col_search:
    busqueda = st.text_input(
        "🔍 Escribe empresa o ticker:",
        placeholder="Ej: Tesla, Apple, FEMSA, Cemex, Bitcoin...",
        key="input_busqueda"
    )

with col_btn:
    st.write("")
    st.write("")
    buscar_btn = st.button("🚀 ANALIZAR", use_container_width=True)

# Autocompletado — muestra sugerencias mientras escribe
if busqueda and len(busqueda) >= 2 and not buscar_btn:
    with st.spinner("Buscando..."):
        sugerencias = buscar_sugerencias(busqueda)

    if sugerencias:
        st.markdown("**Selecciona el activo:**")
        opciones = [s["label"] for s in sugerencias]
        seleccion = st.radio("", opciones, key="radio_sug", label_visibility="collapsed")

        # Guardar selección en session_state
        idx = opciones.index(seleccion)
        st.session_state.ticker_sel = sugerencias[idx]["ticker"]
        st.session_state.nombre_sel = sugerencias[idx]["nombre"]
        st.session_state.pais_sel   = sugerencias[idx]["pais"]

        st.info(f"Seleccionado: **{st.session_state.nombre_sel}** `{st.session_state.ticker_sel}` {st.session_state.pais_sel}")

# Auto-refresh opcional
col_r1, col_r2 = st.columns([1, 5])
with col_r1:
    auto_refresh = st.checkbox("🔄 Auto-refresh 15 min")
if auto_refresh:
    import time
    st.info("🔄 Actualizando cada 15 minutos...")
    time.sleep(900)
    st.rerun()

# ── EJECUTAR ANÁLISIS ──
ticker_a_usar = st.session_state.ticker_sel or busqueda.upper().strip()
nombre_a_usar = st.session_state.nombre_sel or busqueda

if buscar_btn and ticker_a_usar:
    with st.spinner("🤖 AI.Lino analizando con HMM Viterbi..."):
        maquina  = MaquinaDineroLino()
        resultado, error = maquina.analizar(ticker_a_usar)

    if error:
        st.error(f"❌ Error: {error}")
    elif resultado:

        # ── SEÑAL PRINCIPAL ──
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f0c29, #302b63); 
             padding: 30px; border-radius: 15px; text-align: center;
             border: 3px solid {resultado["color"]}; margin: 15px 0;'>
            <h1 style='color: {resultado["color"]}; font-size: 3em;'>{resultado["señal"]}</h1>
            <h2 style='color: white;'>Score AI.LINO: {resultado["score"]}/100</h2>
            <h3 style='color: #aaaaaa;'>💰 Precio en tiempo real: 
            <span style='color: white; font-weight: bold;'>${resultado["precio"]:.2f}</span>
            <span style='color: {"#00ff88" if resultado["cambio"] > 0 else "#ff4444"}'>
            ({resultado["cambio"]:+.2f}%)</span></h3>
        </div>
        """, unsafe_allow_html=True)

        # ── INDICADORES ──
        st.markdown("### 📊 Indicadores")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🧠 HMM Viterbi", resultado["estado_hmm"])
        c2.metric("📈 RSI (14)", resultado["rsi"],
                  "⚠️ Sobrecompra" if resultado["rsi"] > 70 else
                  "✅ Sobreventa"  if resultado["rsi"] < 30 else "Neutral")
        macd_diff = round(resultado["macd"] - resultado["signal"], 4)
        c3.metric("📉 MACD", resultado["macd"], f"vs Signal: {macd_diff:+.4f}")
        c4.metric("📅 Datos históricos", f"{resultado['datos']} días")

        # ── BACKTESTING ──
        st.markdown("### 🔬 Backtesting (2 años)")
        b1, b2, b3 = st.columns(3)
        b1.metric("🟢 Señales Compra", resultado["compras_bt"])
        b2.metric("🔴 Señales Venta",  resultado["ventas_bt"])
        pct = round((resultado["compras_bt"] + resultado["ventas_bt"]) / resultado["total_bt"] * 100, 1)
        b3.metric("⚡ Actividad total", f"{pct}%")

        # ── GRÁFICA PROFESIONAL DE VELAS ──
        st.markdown("### 📈 Gráfica Profesional — Velas Diarias + RSI + MACD")

        fig = grafica_trading_profesional(
            resultado["df_chart"],
            ticker_a_usar,
            nombre_a_usar
        )

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(resultado["close"])

        st.caption(f"🤖 AI.LINO Pro | HMM Viterbi + RSI + MACD | {ticker_a_usar} | Precio en tiempo real vía fast_info")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555;'>🤖 AI.Lino Pro © 2026 — Herramienta educativa. Opera con responsabilidad.</p>",
    unsafe_allow_html=True
)
