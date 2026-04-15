import streamlit as st
import yfinance as yf
import numpy as np
from hmmlearn import hmm
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import gc
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI.Lino Pro", page_icon="🤖", layout="wide")

st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); 
     padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: #00ff88; font-size: 3em; font-family: Arial Black;'>🤖 AI.LINO PRO</h1>
    <h3 style='color: #ffffff;'>Motor de Rebote — HMM + RSI + Volumen + Niveles Exactos</h3>
    <p style='color: #aaaaaa;'>Detecta agotamiento de vendedores · Entrada precisa · Stop Loss automatico</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CACHÉ DE DATOS — evita re-descargar yfinance
# en cada interacción del usuario con la UI.
# TTL=600s swing, TTL=30s precio RT
# ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def descargar_datos(ticker, period, interval):
    try:
        df   = yf.Ticker(ticker).history(period=period, interval=interval)
        cols = [c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]
        return df[cols].copy()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=30, show_spinner=False)
def descargar_precio_rt(ticker):
    try:
        fi    = yf.Ticker(ticker).fast_info
        precio = fi.last_price
        prev   = fi.previous_close
        if precio and prev:
            return float(precio), float(((precio - prev) / prev) * 100)
    except:
        pass
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_sugerencias(texto):
    try:
        resultados = yf.Search(texto, max_results=8)
        sugerencias = []
        for q in resultados.quotes:
            if q.get("quoteType") in ["EQUITY", "ETF", "CRYPTOCURRENCY"]:
                ticker = q.get("symbol", "")
                nombre = q.get("longname") or q.get("shortname", ticker)
                tipo   = q.get("quoteType", "")
                if ticker.endswith(".MX"):
                    pais = "MX"
                elif any(ticker.endswith(x) for x in [".PA",".DE",".AS",".SW",".MC",".MI",".L"]):
                    pais = "EU"
                elif tipo == "CRYPTOCURRENCY":
                    pais = "CRYPTO"
                else:
                    pais = "USA"
                sugerencias.append({
                    "label": f"[{pais}] {nombre} ({ticker})",
                    "ticker": ticker,
                    "nombre": nombre,
                    "pais": pais
                })
        return sugerencias
    except:
        return []

def obtener_precio_realtime(ticker):
    return descargar_precio_rt(ticker)

# ─────────────────────────────────────────────
# INDICADORES
# ─────────────────────────────────────────────
def calcular_rsi(precios, periodo=14):
    delta    = precios.diff()
    ganancia = delta.where(delta > 0, 0).rolling(periodo).mean()
    perdida  = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs       = ganancia / perdida
    return 100 - (100 / (1 + rs))

def calcular_stoch_rsi(precios, periodo=14, smooth=3):
    rsi     = calcular_rsi(precios, periodo)
    rsi_min = rsi.rolling(periodo).min()
    rsi_max = rsi.rolling(periodo).max()
    stoch   = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    k       = stoch.rolling(smooth).mean()
    d       = k.rolling(smooth).mean()
    return k, d

def calcular_macd(precios):
    ema12      = precios.ewm(span=12).mean()
    ema26      = precios.ewm(span=26).mean()
    macd       = ema12 - ema26
    signal     = macd.ewm(span=9).mean()
    histograma = macd - signal
    return macd, signal, histograma

def calcular_bollinger(precios, periodo=20):
    media = precios.rolling(periodo).mean()
    std   = precios.rolling(periodo).std()
    return media + 2*std, media, media - 2*std

# ─────────────────────────────────────────────
# SEMÁFORO DE MOMENTUM DOMINANTE
# ─────────────────────────────────────────────
def calcular_adx(df, periodo=14):
    """Calcula ADX, +DI y -DI para medir fuerza y dirección de tendencia."""
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    close = df['Close'].squeeze()

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up   = high - high.shift()
    down = low.shift() - low

    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    atr_s      = tr.ewm(span=periodo, min_periods=periodo).mean()
    plus_di    = 100 * plus_dm.ewm(span=periodo, min_periods=periodo).mean() / (atr_s + 1e-10)
    minus_di   = 100 * minus_dm.ewm(span=periodo, min_periods=periodo).mean() / (atr_s + 1e-10)
    dx         = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx        = dx.ewm(span=periodo, min_periods=periodo).mean()

    return adx, plus_di, minus_di


def detectar_momentum_dominante(ticker):
    """
    Semáforo de momentum: detecta si hay una fuerza direccional fuerte
    que pueda aplastar un rebote intradía.

    Usa 4 capas:
    1. ADX 15min — fuerza de tendencia (>25 fuerte, >40 muy fuerte)
    2. EMAs en cascada 5min — dirección dominante
    3. Ratio velas rojas/verdes últimas 20 velas
    4. Volumen direccional — dinero detrás de la fuerza

    Retorna: dict con color semáforo, dirección, fuerza, descripción
    """
    try:
        df_15m = descargar_datos(ticker, "5d",  "15m")
        df_5m  = descargar_datos(ticker, "2d",  "5m")

        if df_15m.empty or len(df_15m) < 30:
            return {"error": "Sin datos para semáforo"}

        resultado = {}

        # ── CAPA 1: ADX en 15 minutos ──
        adx_s, pdi_s, mdi_s = calcular_adx(df_15m, periodo=14)
        adx_val  = float(adx_s.iloc[-1])
        pdi_val  = float(pdi_s.iloc[-1])
        mdi_val  = float(mdi_s.iloc[-1])
        adx_prev = float(adx_s.iloc[-4])  # hace ~1 hora

        direccion_adx = "ALCISTA" if pdi_val > mdi_val else "BAJISTA"
        adx_subiendo  = adx_val > adx_prev

        resultado["adx"]           = round(adx_val, 1)
        resultado["pdi"]           = round(pdi_val, 1)
        resultado["mdi"]           = round(mdi_val, 1)
        resultado["adx_subiendo"]  = adx_subiendo
        resultado["direccion_adx"] = direccion_adx

        # ── CAPA 2: EMAs en cascada 5 minutos ──
        if not df_5m.empty and len(df_5m) >= 50:
            close_5m = df_5m['Close'].squeeze()
            ema9_5m  = close_5m.ewm(span=9).mean()
            ema21_5m = close_5m.ewm(span=21).mean()
            ema50_5m = close_5m.ewm(span=50).mean()

            e9  = float(ema9_5m.iloc[-1])
            e21 = float(ema21_5m.iloc[-1])
            e50 = float(ema50_5m.iloc[-1])

            # Pendiente de cada EMA (últimas 6 velas = 30 min)
            pend_e9  = (e9  - float(ema9_5m.iloc[-6]))  / float(ema9_5m.iloc[-6])  * 100
            pend_e21 = (e21 - float(ema21_5m.iloc[-6])) / float(ema21_5m.iloc[-6]) * 100
            pend_e50 = (e50 - float(ema50_5m.iloc[-6])) / float(ema50_5m.iloc[-6]) * 100

            # Cascada perfecta alcista: e9 > e21 > e50 todas subiendo
            cascada_alcista = (e9 > e21 > e50) and (pend_e9 > 0) and (pend_e21 > 0)
            # Cascada perfecta bajista: e9 < e21 < e50 todas bajando
            cascada_bajista = (e9 < e21 < e50) and (pend_e9 < 0) and (pend_e21 < 0)

            resultado["ema9"]             = round(e9, 4)
            resultado["ema21"]            = round(e21, 4)
            resultado["ema50"]            = round(e50, 4)
            resultado["pend_e9"]          = round(pend_e9, 4)
            resultado["cascada_alcista"]  = cascada_alcista
            resultado["cascada_bajista"]  = cascada_bajista
        else:
            cascada_alcista = False
            cascada_bajista = False
            resultado["cascada_alcista"] = False
            resultado["cascada_bajista"] = False

        # ── CAPA 3: Ratio velas rojas vs verdes (últimas 20 velas 5min) ──
        if not df_5m.empty and len(df_5m) >= 20:
            ultimas_20 = df_5m.iloc[-20:]
            verdes = sum(1 for c, o in zip(ultimas_20['Close'], ultimas_20['Open']) if c >= o)
            rojas  = 20 - verdes
            ratio_direccional = verdes / 20  # 1.0 = todo verde, 0.0 = todo rojo

            resultado["velas_verdes"] = verdes
            resultado["velas_rojas"]  = rojas
            resultado["ratio_dir"]    = round(ratio_direccional, 2)
        else:
            ratio_direccional = 0.5
            resultado["velas_verdes"] = 10
            resultado["velas_rojas"]  = 10
            resultado["ratio_dir"]    = 0.5

        # ── CAPA 4: Volumen direccional ──
        if not df_5m.empty and len(df_5m) >= 20:
            ultimas = df_5m.iloc[-20:]
            vol_verde = sum(
                float(v) for c, o, v in zip(ultimas['Close'], ultimas['Open'], ultimas['Volume'])
                if c >= o
            )
            vol_rojo = sum(
                float(v) for c, o, v in zip(ultimas['Close'], ultimas['Open'], ultimas['Volume'])
                if c < o
            )
            total_vol = vol_verde + vol_rojo + 1e-10
            ratio_vol_dir = vol_verde / total_vol  # >0.6 = fuerza alcista con volumen

            resultado["vol_verde_pct"] = round(vol_verde / total_vol * 100, 1)
            resultado["vol_rojo_pct"]  = round(vol_rojo  / total_vol * 100, 1)
            resultado["ratio_vol_dir"] = round(ratio_vol_dir, 2)
        else:
            ratio_vol_dir = 0.5
            resultado["vol_verde_pct"] = 50
            resultado["vol_rojo_pct"]  = 50
            resultado["ratio_vol_dir"] = 0.5

        # ── DECISIÓN DEL SEMÁFORO ──
        # Puntaje de fuerza alcista (0-100)
        score_alcista = 0
        score_bajista = 0

        # ADX aporta fuerza en la dirección que apunta
        if adx_val > 40:
            fuerza_adx = 40
        elif adx_val > 25:
            fuerza_adx = 25
        elif adx_val > 15:
            fuerza_adx = 10
        else:
            fuerza_adx = 0

        if direccion_adx == "ALCISTA":
            score_alcista += fuerza_adx
        else:
            score_bajista += fuerza_adx

        # EMAs en cascada
        if cascada_alcista:
            score_alcista += 25
        elif cascada_bajista:
            score_bajista += 25

        # Ratio velas
        if ratio_direccional > 0.70:
            score_alcista += 20
        elif ratio_direccional < 0.30:
            score_bajista += 20
        elif ratio_direccional > 0.60:
            score_alcista += 10
        elif ratio_direccional < 0.40:
            score_bajista += 10

        # Volumen direccional
        if ratio_vol_dir > 0.65:
            score_alcista += 15
        elif ratio_vol_dir < 0.35:
            score_bajista += 15

        score_neto = score_alcista - score_bajista  # positivo = alcista, negativo = bajista

        resultado["score_alcista"] = score_alcista
        resultado["score_bajista"] = score_bajista
        resultado["score_neto"]    = score_neto

        # ── SEMÁFORO FINAL ──
        adx_fuerte = adx_val > 25

        if abs(score_neto) < 20 and adx_val < 20:
            # Sin fuerza dominante — ideal para rebotes
            semaforo       = "VERDE"
            color_sem      = "#00ff88"
            icono          = "🟢"
            titulo         = "SIN FUERZA DOMINANTE"
            descripcion    = "Mercado en rango/lateral — condiciones ideales para rebote"
            recomendacion  = "✅ Puedes operar el rebote con confianza"
            impacto_rebote = "ALTO"

        elif abs(score_neto) < 35 and adx_val < 30:
            # Fuerza moderada
            if score_neto > 0:
                dir_txt = "ALCISTA moderada"
                rec_txt = "✅ Fuerza alcista ayuda al rebote — buenas condiciones"
                impacto = "FAVORABLE"
            else:
                dir_txt = "BAJISTA moderada"
                rec_txt = "⚠️ Fuerza bajista presente — rebote posible pero será breve"
                impacto = "LIMITADO"
            semaforo       = "AMARILLO"
            color_sem      = "#FFD700"
            icono          = "🟡"
            titulo         = f"FUERZA {dir_txt.upper()}"
            descripcion    = f"ADX {adx_val:.0f} — tendencia {dir_txt}"
            recomendacion  = rec_txt
            impacto_rebote = impacto

        else:
            # Fuerza dominante fuerte
            if score_neto > 0:
                dir_txt = "ALCISTA"
                rec_txt = "✅ Fuerza alcista dominante — rebote tiene respaldo, seguir tendencia"
                impacto = "MUY FAVORABLE"
                color_sem = "#00BFFF"
                icono     = "🔵"
            else:
                dir_txt = "BAJISTA"
                rec_txt = "🚨 Fuerza bajista dominante — rebote será aplastado, NO entrar"
                impacto = "BLOQUEADO"
                color_sem = "#ff4444"
                icono     = "🔴"
            semaforo       = "ROJO" if score_neto < 0 else "AZUL"
            titulo         = f"FUERZA DOMINANTE {dir_txt}"
            descripcion    = (
                f"ADX {adx_val:.0f} {'↑ subiendo' if adx_subiendo else '↓ bajando'} — "
                f"+DI {pdi_val:.0f} vs -DI {mdi_val:.0f}"
            )
            recomendacion  = rec_txt
            impacto_rebote = impacto

        resultado["semaforo"]       = semaforo
        resultado["color_sem"]      = color_sem
        resultado["icono"]          = icono
        resultado["titulo"]         = titulo
        resultado["descripcion"]    = descripcion
        resultado["recomendacion"]  = recomendacion
        resultado["impacto_rebote"] = impacto_rebote

        return resultado

    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# ██████████████████████████████████████████████
# MÓDULO NUEVO: DETECTOR DE PISO INTRADÍA
# ██████████████████████████████████████████████
# ─────────────────────────────────────────────
def detectar_piso_intraday(ticker):
    """
    Detecta el piso intradía con contexto histórico de 10 días.

    Capas:
    A) CONTEXTO HISTÓRICO — ¿Vale la pena entrar hoy?
       - Posición del precio en el rango de 10 días (techo/piso/medio)
       - Rango promedio diario real vs caída de hoy
       - Velocidad de caída (¿caída libre o corrección normal?)
       - Soportes reales de múltiples días
    B) SEÑALES INTRADÍA — ¿El piso ya se está formando?
       - Caída + volumen decreciente
       - Velas de rechazo en 5min
       - Stoch RSI en sobreventa
       - Compresión de rangos

    Retorna: dict completo con contexto + señal + niveles
    """
    try:
        t = yf.Ticker(ticker)

        # ── DATOS MULTI-TIMEFRAME ──
        df_5m  = descargar_datos(ticker, "5d",  "5m")
        df_15m = descargar_datos(ticker, "10d", "15m")
        df_1h  = descargar_datos(ticker, "20d", "1h")
        df_1d  = descargar_datos(ticker, "20d", "1d")

        if df_5m.empty or len(df_5m) < 20:
            return {"error": "Sin datos intradía suficientes"}

        close_5m  = df_5m['Close'].squeeze()
        volume_5m = df_5m['Volume'].squeeze()
        high_5m   = df_5m['High'].squeeze()
        low_5m    = df_5m['Low'].squeeze()
        open_5m   = df_5m['Open'].squeeze()

        precio_actual = float(close_5m.iloc[-1])
        puntos        = 0
        señales       = []
        alertas_id    = []
        contexto      = {}   # Dict para mostrar en UI

        # ════════════════════════════════════════════════
        # BLOQUE A — CONTEXTO HISTÓRICO (multi-día)
        # ════════════════════════════════════════════════

        # ── A1: Rango promedio diario real (últimos 10 días) ──
        # Calcula cuánto se mueve NORMALMENTE esta acción en un día
        rangos_diarios = []
        if not df_1d.empty and len(df_1d) >= 5:
            close_1d = df_1d['Close'].squeeze()
            high_1d  = df_1d['High'].squeeze()
            low_1d   = df_1d['Low'].squeeze()
            for i in range(-min(10, len(df_1d)), 0):
                rango_d = float(high_1d.iloc[i]) - float(low_1d.iloc[i])
                rangos_diarios.append(rango_d)
            rango_promedio_diario = sum(rangos_diarios) / len(rangos_diarios)
        else:
            rango_promedio_diario = precio_actual * 0.015  # fallback 1.5%

        contexto["rango_prom_diario"] = round(rango_promedio_diario, 2)

        # ── A2: ¿Cuánto ha caído hoy vs el promedio? ──
        # Si hoy cayó más del promedio = caída inusual/fuerte
        # Si cayó menos = corrección normal, rebote más probable
        precio_apertura_hoy = None
        if not df_5m.empty:
            # Detectar apertura de la sesión de hoy (primeras velas del día)
            hoy = pd.Timestamp.now().date()
            velas_hoy = df_5m[df_5m.index.date == hoy]
            if not velas_hoy.empty:
                precio_apertura_hoy = float(velas_hoy['Open'].iloc[0])
                min_hoy  = float(velas_hoy['Low'].squeeze().min())
                max_hoy  = float(velas_hoy['High'].squeeze().max())
                caida_hoy = precio_apertura_hoy - precio_actual
                caida_pct = (caida_hoy / precio_apertura_hoy) * 100 if precio_apertura_hoy > 0 else 0
            else:
                # Usar últimas ~78 velas de 5min = ~1 sesión completa
                min_hoy  = float(low_5m.iloc[-78:].min())
                max_hoy  = float(high_5m.iloc[-78:].max())
                caida_hoy = 0
                caida_pct = 0
        else:
            min_hoy  = float(low_5m.min())
            max_hoy  = float(high_5m.max())
            caida_hoy = 0
            caida_pct = 0

        contexto["min_hoy"]     = round(min_hoy, 2)
        contexto["max_hoy"]     = round(max_hoy, 2)
        contexto["caida_hoy"]   = round(caida_hoy, 2)
        contexto["caida_pct"]   = round(caida_pct, 2)

        ratio_caida = (caida_hoy / rango_promedio_diario) if rango_promedio_diario > 0 else 0
        contexto["ratio_caida"] = round(ratio_caida, 2)

        if ratio_caida > 1.5:
            alertas_id.append(
                f"🚨 Caída de hoy (${caida_hoy:.2f}) es {ratio_caida:.1f}x el rango normal "
                f"(${rango_promedio_diario:.2f}) — posible caída libre, NO entrar aún"
            )
            puntos -= 3  # Penalización fuerte
        elif ratio_caida > 1.0:
            alertas_id.append(
                f"⚠️ Caída de hoy supera el rango promedio ({ratio_caida:.1f}x) — cautela"
            )
            puntos -= 1
        elif ratio_caida > 0.3:
            puntos += 2
            señales.append(
                f"✅ Caída de hoy ({caida_pct:.1f}%) dentro del rango normal — corrección sana"
            )
        else:
            alertas_id.append("ℹ️ Precio apenas ha bajado hoy — esperar corrección mayor")

        # ── A3: Posición del precio en el rango de 10 días ──
        # ¿Está en el techo, en el piso, o en el medio?
        if not df_15m.empty and len(df_15m) >= 20:
            high_10d = float(df_15m['High'].squeeze().max())
            low_10d  = float(df_15m['Low'].squeeze().min())
            rango_10d = high_10d - low_10d

            if rango_10d > 0:
                posicion_pct = ((precio_actual - low_10d) / rango_10d) * 100
                contexto["posicion_10d_pct"] = round(posicion_pct, 1)
                contexto["high_10d"]         = round(high_10d, 2)
                contexto["low_10d"]          = round(low_10d, 2)

                if posicion_pct <= 20:
                    puntos += 4
                    señales.append(
                        f"🎯 Precio en el PISO del rango 10 días ({posicion_pct:.0f}%) "
                        f"— zona de máximo valor histórico reciente"
                    )
                elif posicion_pct <= 35:
                    puntos += 2
                    señales.append(
                        f"📍 Precio en zona baja del rango 10 días ({posicion_pct:.0f}%) "
                        f"— buena zona de entrada"
                    )
                elif posicion_pct >= 80:
                    alertas_id.append(
                        f"🚫 Precio en el TECHO del rango 10 días ({posicion_pct:.0f}%) "
                        f"— NO es zona de compra, esperar corrección"
                    )
                    puntos -= 4  # Penalización fuerte por comprar en techo
                elif posicion_pct >= 65:
                    alertas_id.append(
                        f"⚠️ Precio en zona alta del rango 10 días ({posicion_pct:.0f}%) "
                        f"— riesgo elevado de entrar aquí"
                    )
                    puntos -= 2
                else:
                    señales.append(
                        f"➡️ Precio en zona media del rango ({posicion_pct:.0f}%) "
                        f"— esperar que baje más para mejor entrada"
                    )
        else:
            posicion_pct = 50
            contexto["posicion_10d_pct"] = 50
            contexto["high_10d"] = max_hoy
            contexto["low_10d"]  = min_hoy

        # ── A4: Velocidad de caída (¿caída libre o freno?) ──
        # Compara velocidad de caída en últimas 6 velas 15min vs anteriores
        if not df_15m.empty and len(df_15m) >= 20:
            close_15m = df_15m['Close'].squeeze()
            # Velocidad = cambio promedio por vela
            vel_rec = abs(float(close_15m.iloc[-6:].diff().mean()))
            vel_ant = abs(float(close_15m.iloc[-18:-6].diff().mean()))
            if vel_ant > 0:
                ratio_vel = vel_rec / vel_ant
                contexto["ratio_velocidad"] = round(ratio_vel, 2)
                if ratio_vel > 2.0:
                    alertas_id.append(
                        f"🚨 Velocidad de caída acelerada ({ratio_vel:.1f}x normal) "
                        f"— caída libre, esperar estabilización"
                    )
                    puntos -= 2
                elif ratio_vel < 0.5:
                    puntos += 2
                    señales.append(
                        f"🐢 Caída desacelerando ({ratio_vel:.1f}x) "
                        f"— momentum bajista perdiendo fuerza"
                    )
                else:
                    contexto["ratio_velocidad"] = round(ratio_vel, 2)

        # ── A5: Soportes reales de múltiples días ──
        # Identifica niveles donde el precio rebotó en los últimos 10 días
        soportes_reales = []
        if not df_1h.empty and len(df_1h) >= 10:
            low_1h  = df_1h['Low'].squeeze()
            high_1h = df_1h['High'].squeeze()

            # Soporte fuerte = mínimo que se tocó 2+ veces sin romperse
            min_10d_abs = float(low_1h.min())
            min_5d      = float(low_1h.iloc[-5*8:].min())   # ~5 días en 1h
            min_3d      = float(low_1h.iloc[-3*8:].min())   # ~3 días
            percentil_10 = float(np.percentile(low_1h, 10))  # 10% más bajo

            # Soporte más cercano por abajo del precio actual
            candidatos = [min_10d_abs, min_5d, min_3d, percentil_10]
            for s in candidatos:
                if s < precio_actual * 1.02:  # Dentro del 2% por debajo
                    soportes_reales.append(round(s, 2))

            soportes_reales = sorted(list(set(soportes_reales)), reverse=True)
            contexto["soportes_reales"] = soportes_reales[:3]

            # ¿El precio actual está cerca de un soporte real?
            for s in soportes_reales:
                dist_pct = ((precio_actual - s) / precio_actual) * 100
                if dist_pct <= 0.5:
                    puntos += 4
                    señales.append(
                        f"🧱 Precio EN soporte histórico multi-día (${s:.2f}) "
                        f"— nivel testado previamente"
                    )
                    break
                elif dist_pct <= 1.5:
                    puntos += 2
                    señales.append(
                        f"📌 Precio cerca de soporte histórico (${s:.2f}, -{dist_pct:.1f}%) "
                        f"— zona de interés"
                    )
                    break
        else:
            contexto["soportes_reales"] = [round(min_hoy, 2)]

        # ── A6: ¿Rebotó desde este nivel antes? (validación de soporte) ──
        if not df_1h.empty and len(df_1h) >= 20 and soportes_reales:
            low_1h   = df_1h['Low'].squeeze()
            close_1h = df_1h['Close'].squeeze()
            soporte_ref = soportes_reales[0]
            tolerancia  = soporte_ref * 0.015
            rebotes_confirmados = 0
            for i in range(len(df_1h) - 1):
                if abs(float(low_1h.iloc[i]) - soporte_ref) <= tolerancia:
                    if float(close_1h.iloc[i+1]) > float(close_1h.iloc[i]):
                        rebotes_confirmados += 1
            if rebotes_confirmados >= 2:
                puntos += 3
                señales.append(
                    f"🔁 Soporte ${soporte_ref:.2f} confirmado con "
                    f"{rebotes_confirmados} rebotes previos — nivel validado"
                )
            elif rebotes_confirmados == 1:
                puntos += 1
                señales.append(
                    f"🔁 Soporte ${soporte_ref:.2f} con 1 rebote previo — nivel a vigilar"
                )

        # ════════════════════════════════════════════════
        # BLOQUE B — SEÑALES INTRADÍA (5 min)
        # ════════════════════════════════════════════════

        # ── B1: Caída de precio + caída de volumen ──
        bajistas_consecutivas = 0
        vol_decreciente       = True
        for i in range(-6, -1):
            if float(close_5m.iloc[i]) < float(close_5m.iloc[i-1]):
                bajistas_consecutivas += 1
                if i < -2 and float(volume_5m.iloc[i]) > float(volume_5m.iloc[i-1]):
                    vol_decreciente = False
            else:
                bajistas_consecutivas = 0
                vol_decreciente       = True

        if bajistas_consecutivas >= 3 and vol_decreciente:
            puntos += 3
            señales.append(
                f"📉 Caída {bajistas_consecutivas} velas con volumen decreciente "
                f"— vendedores agotándose en 5min"
            )
        elif bajistas_consecutivas >= 2:
            puntos += 1
            señales.append(f"⚠️ Presión bajista ({bajistas_consecutivas} velas) — vigilar")

        vol_rec_5 = float(volume_5m.iloc[-5:].mean())
        vol_ant_5 = float(volume_5m.iloc[-15:-5].mean())
        if vol_ant_5 > 0 and vol_rec_5 < vol_ant_5 * 0.70:
            reduccion_vol = ((vol_ant_5 - vol_rec_5) / vol_ant_5) * 100
            puntos += 2
            señales.append(
                f"📉 Volumen cayó {reduccion_vol:.0f}% en últimas 5 velas — vendedores secándose"
            )

        # ── B2: Vela de rechazo ──
        rechazos_5m = 0
        for i in range(-4, 0):
            o = float(open_5m.iloc[i])
            h = float(high_5m.iloc[i])
            l = float(low_5m.iloc[i])
            c = float(close_5m.iloc[i])
            cuerpo    = abs(c - o) + 1e-10
            mecha_inf = min(o, c) - l
            rango     = h - l + 1e-10
            if mecha_inf > cuerpo * 1.8 and mecha_inf / rango > 0.35:
                rechazos_5m += 1
            elif cuerpo / rango < 0.15 and rango > 0:
                rechazos_5m += 0.5

        if rechazos_5m >= 2:
            puntos += 3
            señales.append(
                f"🕯️ {int(rechazos_5m)} velas de rechazo — compradores defendiendo el piso"
            )
        elif rechazos_5m >= 1:
            puntos += 2
            señales.append("🕯️ Vela de rechazo detectada — posible piso")

        # ── B3: Stoch RSI intradía ──
        if len(close_5m) >= 20:
            stoch_k_5m, stoch_d_5m = calcular_stoch_rsi(close_5m, periodo=14, smooth=3)
            sk_actual = float(stoch_k_5m.iloc[-1])
            sd_actual = float(stoch_d_5m.iloc[-1])
            if sk_actual < 10:
                puntos += 3
                señales.append(
                    f"📊 Stoch RSI 5min extremo ({sk_actual:.1f}) — rebote inminente"
                )
            elif sk_actual < 20:
                puntos += 2
                señales.append(
                    f"📊 Stoch RSI 5min en sobreventa ({sk_actual:.1f}) — zona de rebote"
                )
            elif sk_actual < 30:
                puntos += 1
                señales.append(
                    f"📊 Stoch RSI bajando ({sk_actual:.1f}) — acercándose a zona rebote"
                )
            if sk_actual > sd_actual and sk_actual < 40:
                puntos += 1
                señales.append("📈 Stoch RSI cruzando al alza — impulso iniciando")
        else:
            sk_actual = 50
            sd_actual = 50

        # ── B4: Compresión de rangos 5min ──
        rangos_rec = [float(high_5m.iloc[i]) - float(low_5m.iloc[i]) for i in range(-4, 0)]
        rangos_ant = [float(high_5m.iloc[i]) - float(low_5m.iloc[i]) for i in range(-8, -4)]
        if rangos_ant and sum(rangos_ant) > 0:
            rango_rec_avg = sum(rangos_rec) / len(rangos_rec)
            rango_ant_avg = sum(rangos_ant) / len(rangos_ant)
            if rango_rec_avg < rango_ant_avg * 0.75:
                compresion = ((rango_ant_avg - rango_rec_avg) / rango_ant_avg) * 100
                puntos += 2
                señales.append(
                    f"🔇 Rango comprimido {compresion:.0f}% — la caída pierde velocidad"
                )

        # ── NIVELES DE OPERACIÓN (basados en contexto histórico) ──
        atr_5m    = float((high_5m - low_5m).rolling(14).mean().iloc[-1])
        soporte_1h = contexto.get("soportes_reales", [precio_actual * 0.99])[0] \
                     if contexto.get("soportes_reales") else precio_actual * 0.99

        entrada   = round(precio_actual, 2)
        # Stop bajo el soporte histórico más cercano, no solo el mínimo de hoy
        stop_loss = round(soporte_1h - atr_5m * 0.8, 2)
        objetivo1 = round(precio_actual + atr_5m * 2.5, 2)
        objetivo2 = round(precio_actual + atr_5m * 5.0, 2)
        riesgo    = round(((precio_actual - stop_loss) / precio_actual) * 100, 2)
        ganancia  = round(((objetivo1 - precio_actual) / precio_actual) * 100, 2)
        rr        = round(ganancia / riesgo, 1) if riesgo > 0 else 0

        # ── ALERTAS ADICIONALES ──
        rsi_5m = float(calcular_rsi(close_5m).iloc[-1])
        if rsi_5m > 60:
            alertas_id.append(
                f"⚠️ RSI 5min elevado ({rsi_5m:.1f}) — precio sobrecomprado en intradía"
            )
        if bajistas_consecutivas == 0 and ratio_caida < 0.2:
            alertas_id.append(
                "ℹ️ El precio no ha bajado suficiente hoy — esperar corrección antes de entrar"
            )

        # ── DECISIÓN FINAL con penalizaciones aplicadas ──
        if puntos >= 12:
            nivel     = "FUERTE"
            color     = "#00ff88"
            señal_txt = "🟢 POSIBLE PISO — ENTRAR AHORA"
            confianza = min(95, 55 + puntos * 2)
        elif puntos >= 8:
            nivel     = "MODERADO"
            color     = "#FFD700"
            señal_txt = "🟡 PISO PROBABLE — PREPARARSE"
            confianza = min(78, 40 + puntos * 2)
        elif puntos >= 4:
            nivel     = "DEBIL"
            color     = "#FF8C00"
            señal_txt = "🟠 PRIMERAS SEÑALES — VIGILAR"
            confianza = min(55, 25 + puntos * 2)
        else:
            nivel     = "SIN SEÑAL"
            color     = "#ff4444"
            señal_txt = "🔴 NO ENTRAR — CONDICIONES DESFAVORABLES"
            confianza = max(5, puntos * 4)

        return {
            "señal":              señal_txt,
            "nivel":              nivel,
            "color":              color,
            "puntos":             puntos,
            "confianza":          confianza,
            "precio_actual":      precio_actual,
            "min_hoy":            contexto.get("min_hoy", min_hoy),
            "max_hoy":            contexto.get("max_hoy", max_hoy),
            "high_10d":           contexto.get("high_10d", max_hoy),
            "low_10d":            contexto.get("low_10d",  min_hoy),
            "posicion_10d_pct":   contexto.get("posicion_10d_pct", 50),
            "rango_prom_diario":  contexto.get("rango_prom_diario", 0),
            "caida_hoy":          contexto.get("caida_hoy", 0),
            "caida_pct":          contexto.get("caida_pct", 0),
            "ratio_caida":        contexto.get("ratio_caida", 0),
            "ratio_velocidad":    contexto.get("ratio_velocidad", 1),
            "soportes_reales":    contexto.get("soportes_reales", []),
            "soporte_1h":         soporte_1h,
            "resistencia_1h":     contexto.get("high_10d", max_hoy),
            "entrada":            entrada,
            "stop_loss":          stop_loss,
            "objetivo1":          objetivo1,
            "objetivo2":          objetivo2,
            "riesgo_pct":         riesgo,
            "ganancia_pct":       ganancia,
            "rr":                 rr,
            "atr_5m":             round(atr_5m, 4),
            "stoch_k_5m":         round(sk_actual, 1),
            "rsi_5m":             round(rsi_5m, 1),
            "señales":            señales,
            "alertas":            alertas_id,
            "df_5m":              df_5m,
            "df_1h":              df_1h,
        }

    except Exception as e:
        return {"error": str(e)}


def grafica_intraday(df_5m, df_1h, niveles_id, ticker, nombre):
    """Gráfica intradía de 5 minutos con señales de piso."""
    if df_5m is None or df_5m.empty or len(df_5m) < 10:
        return None

    close_5m           = df_5m['Close'].squeeze()
    stoch_k, stoch_d   = calcular_stoch_rsi(close_5m)
    vol_ma             = df_5m['Volume'].squeeze().rolling(20).mean()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=(
            f"Intradía 5min — {nombre} ({ticker})",
            "Volumen 5min",
            "Stoch RSI 5min — Señal de Piso",
        )
    )

    # Velas 5 min
    fig.add_trace(go.Candlestick(
        x=df_5m.index,
        open=df_5m['Open'], high=df_5m['High'],
        low=df_5m['Low'],   close=df_5m['Close'],
        name="5min",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88',  decreasing_fillcolor='#ff4444',
    ), row=1, col=1)

    # Niveles horizontales
    px_ini = df_5m.index[max(0, len(df_5m)-50)]
    px_fin = df_5m.index[-1]
    for y_val, color, etq in [
        (niveles_id.get('entrada'),    '#00ff88', f"Entrada ${niveles_id.get('entrada', 0):.2f}"),
        (niveles_id.get('stop_loss'),  '#ff4444', f"Stop ${niveles_id.get('stop_loss', 0):.2f}"),
        (niveles_id.get('objetivo1'),  '#FFD700', f"T1 ${niveles_id.get('objetivo1', 0):.2f}"),
        (niveles_id.get('objetivo2'),  '#00BFFF', f"T2 ${niveles_id.get('objetivo2', 0):.2f}"),
        (niveles_id.get('min_dia'),    '#FF8C00', f"Mín día ${niveles_id.get('min_dia', 0):.2f}"),
    ]:
        if y_val:
            fig.add_shape(type="line",
                x0=px_ini, x1=px_fin, y0=y_val, y1=y_val,
                line=dict(color=color, width=1.5, dash="dash"), row=1, col=1)
            fig.add_annotation(x=px_fin, y=y_val, text=etq,
                showarrow=False, xanchor="right",
                font=dict(color=color, size=10),
                bgcolor="rgba(0,0,0,0.6)", row=1, col=1)

    # Soporte 1h
    if niveles_id.get('soporte_1h'):
        fig.add_shape(type="line",
            x0=px_ini, x1=px_fin,
            y0=niveles_id['soporte_1h'], y1=niveles_id['soporte_1h'],
            line=dict(color='#9B59B6', width=2, dash="dot"), row=1, col=1)
        fig.add_annotation(
            x=px_ini, y=niveles_id['soporte_1h'],
            text=f"Soporte 1h ${niveles_id['soporte_1h']:.2f}",
            showarrow=False, xanchor="left",
            font=dict(color='#9B59B6', size=9),
            bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    # Volumen
    colores_vol = ['#00ff88' if c >= o else '#ff4444'
                   for c, o in zip(df_5m['Close'], df_5m['Open'])]
    fig.add_trace(go.Bar(x=df_5m.index, y=df_5m['Volume'],
        name="Volumen", marker_color=colores_vol, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=vol_ma,
        name="Vol MA20", line=dict(color='#888888', width=1, dash='dot')), row=2, col=1)

    # Stoch RSI
    fig.add_trace(go.Scatter(x=df_5m.index, y=stoch_k,
        name="Stoch K", line=dict(color='#00ff88', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_5m.index, y=stoch_d,
        name="Stoch D", line=dict(color='#ff4444', width=1.5, dash='dot')), row=3, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="#00ff88", opacity=0.10, line_width=0, row=3, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ff4444", opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#00ff88", line_width=1, row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ff4444", line_width=1, row=3, col=1)

    fig.update_layout(
        height=700,
        paper_bgcolor='#0f0c29', plot_bgcolor='#0f0c29',
        font=dict(color='#ffffff', size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor='rgba(0,0,0,0.4)', font=dict(size=9)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         showspikes=True, spikecolor="#00ff88",
                         spikethickness=1, spikemode="across", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         tickfont=dict(size=9), row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig

# ─────────────────────────────────────────────
# MOTOR DE AGOTAMIENTO DE VENDEDORES (5 CAPAS)
# ─────────────────────────────────────────────
def detectar_agotamiento_vendedores(df):
    if len(df) < 15:
        return False, 0, "Sin datos suficientes", []

    close  = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()

    puntos   = 0
    detalles = []

    for v in [3, 5, 8]:
        if len(df) >= v * 2:
            p_rec   = close.iloc[-v:].mean()
            p_ant   = close.iloc[-v*2:-v].mean()
            vol_rec = volume.iloc[-v:].mean()
            vol_ant = volume.iloc[-v*2:-v].mean()
            if p_rec < p_ant and vol_rec < vol_ant:
                reduccion = ((vol_ant - vol_rec) / vol_ant) * 100
                puntos += 1
                detalles.append(f"Volumen cae {reduccion:.0f}% mientras precio baja (ventana {v}d)")

    rechazos = 0
    for i in range(-5, 0):
        o = float(df['Open'].iloc[i])
        h = float(df['High'].iloc[i])
        l = float(df['Low'].iloc[i])
        c = float(df['Close'].iloc[i])
        cuerpo    = abs(c - o) + 1e-10
        mecha_inf = min(o, c) - l
        rango     = h - l + 1e-10
        if mecha_inf > cuerpo * 1.5 and mecha_inf / rango > 0.35:
            rechazos += 1
    if rechazos >= 2:
        puntos += 2
        detalles.append(f"{rechazos} velas con rechazo de minimos — compradores absorbiendo presion")
    elif rechazos == 1:
        puntos += 1
        detalles.append("1 vela con rechazo de minimos detectada")

    vols_bajas = []
    for i in range(-8, 0):
        if float(close.iloc[i]) < float(df['Open'].iloc[i]):
            vols_bajas.append(float(volume.iloc[i]))
    if len(vols_bajas) >= 3:
        mitad        = len(vols_bajas) // 2
        vol_reciente = sum(vols_bajas[mitad:]) / max(len(vols_bajas[mitad:]), 1)
        vol_anterior = sum(vols_bajas[:mitad]) / max(mitad, 1)
        if vol_reciente < vol_anterior * 0.85:
            reduccion = ((vol_anterior - vol_reciente) / vol_anterior) * 100
            puntos += 2
            detalles.append(f"Dias bajistas con volumen {reduccion:.0f}% menor — vendedores agotandose")

    rangos_rec = [float(high.iloc[i]) - float(low.iloc[i]) for i in range(-5, 0)]
    rangos_ant = [float(high.iloc[i]) - float(low.iloc[i]) for i in range(-10, -5)]
    if rangos_ant and sum(rangos_ant) > 0:
        rango_rec = sum(rangos_rec) / len(rangos_rec)
        rango_ant = sum(rangos_ant) / len(rangos_ant)
        if rango_rec < rango_ant * 0.80:
            compresion = ((rango_ant - rango_rec) / rango_ant) * 100
            puntos += 1
            detalles.append(f"Rango diario comprimido {compresion:.0f}% — la caida pierde velocidad")

    min_rec = float(low.iloc[-5:].min())
    min_ant = float(low.iloc[-15:-5].min())
    if min_ant > 0:
        tolerancia = min_ant * 0.015
        if abs(min_rec - min_ant) <= tolerancia:
            puntos += 3
            detalles.append(f"Doble piso en ${min_rec:.2f} — soporte fuerte confirmado en 2 toques")

    if puntos >= 6:
        return True, 3, "AGOTAMIENTO FUERTE — Rebote de alta probabilidad", detalles
    elif puntos >= 4:
        return True, 2, "AGOTAMIENTO MODERADO — Señales positivas acumulandose", detalles
    elif puntos >= 2:
        return True, 1, "AGOTAMIENTO DEBIL — Primeras señales visibles", detalles
    else:
        return False, 0, "Sin agotamiento confirmado aun", detalles


def detectar_vela_rebote(df):
    senales = []
    if len(df) < 2:
        return senales
    for i in range(1, len(df)):
        o = float(df['Open'].iloc[i])
        h = float(df['High'].iloc[i])
        l = float(df['Low'].iloc[i])
        c = float(df['Close'].iloc[i])
        cuerpo      = abs(c - o)
        rango_total = h - l + 1e-10
        mecha_inf   = min(o, c) - l
        mecha_sup   = h - max(o, c)
        if mecha_inf >= 2 * cuerpo and mecha_sup <= cuerpo * 0.5 and cuerpo / rango_total < 0.4:
            senales.append((df.index[i], "Hammer", "#00ff88"))
        elif cuerpo / rango_total < 0.1:
            senales.append((df.index[i], "Doji", "#FFD700"))
        elif i > 0:
            o_prev = float(df['Open'].iloc[i-1])
            c_prev = float(df['Close'].iloc[i-1])
            if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
                senales.append((df.index[i], "Engulfing Alcista", "#00BFFF"))
    return senales

def calcular_niveles_trading(df, precio_actual):
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr              = tr.rolling(14).mean().iloc[-1]
    soporte_reciente = low.iloc[-10:].min()
    resistencia      = high.iloc[-10:].max()
    soporte_fuerte   = low.iloc[-30:].min()
    _, _, bb_low_s   = calcular_bollinger(close)
    entrada_ideal    = soporte_reciente * 1.005
    stop_loss        = soporte_reciente - atr
    objetivo_1       = precio_actual + atr * 1.5
    objetivo_2       = precio_actual + atr * 3.0
    objetivo_3       = resistencia * 0.995
    riesgo_pct       = ((precio_actual - stop_loss) / precio_actual) * 100
    pot_ganancia     = ((objetivo_2 - precio_actual) / precio_actual) * 100
    rr_ratio         = pot_ganancia / riesgo_pct if riesgo_pct > 0 else 0
    return {
        "entrada":        round(entrada_ideal, 4),
        "stop_loss":      round(stop_loss, 4),
        "objetivo_1":     round(objetivo_1, 4),
        "objetivo_2":     round(objetivo_2, 4),
        "objetivo_3":     round(objetivo_3, 4),
        "soporte":        round(soporte_reciente, 4),
        "resistencia":    round(resistencia, 4),
        "soporte_fuerte": round(soporte_fuerte, 4),
        "atr":            round(atr, 4),
        "riesgo_pct":     round(riesgo_pct, 2),
        "pot_ganancia":   round(pot_ganancia, 2),
        "rr_ratio":       round(rr_ratio, 2),
        "bb_low":         round(bb_low_s.iloc[-1], 4),
    }

def calcular_score_rebote(estado_hmm, rsi, stoch_k, stoch_d,
                           macd_val, signal_val, agot_nivel,
                           velas_rebote, precio_vs_bb_low):
    score   = 50
    razones = []
    alertas = []

    if estado_hmm == "ALCISTA":
        score += 15
        razones.append("HMM detecta regimen alcista")
    elif estado_hmm == "BAJISTA":
        score -= 10
        alertas.append("HMM bajista — rebote de corta duracion posible")
    else:
        razones.append("HMM lateral — posible acumulacion")

    if rsi < 25:
        score += 25
        razones.append(f"RSI extremo ({rsi:.1f}) — sobreventa severa")
    elif rsi < 35:
        score += 15
        razones.append(f"RSI en sobreventa ({rsi:.1f}) — zona de rebote")
    elif rsi > 70:
        score -= 20
        alertas.append(f"RSI sobrecomprado ({rsi:.1f}) — no entrar")
    elif rsi > 60:
        score -= 10
        alertas.append(f"RSI elevado ({rsi:.1f}) — cuidado")

    if stoch_k < 20 and stoch_d < 20:
        score += 20
        razones.append(f"Stoch RSI extremo ({stoch_k:.1f}) — rebote inminente")
    elif stoch_k > stoch_d and stoch_k < 40:
        score += 10
        razones.append(f"Stoch RSI cruzando alza ({stoch_k:.1f}) — impulso iniciando")
    elif stoch_k > 80:
        score -= 15
        alertas.append(f"Stoch RSI sobrecomprado ({stoch_k:.1f}) — salir pronto")

    if macd_val > signal_val:
        score += 10
        razones.append("MACD sobre senal — momentum positivo")
    else:
        score -= 5
        alertas.append("MACD bajo senal — presion vendedora activa")

    if agot_nivel == 3:
        score += 25
        razones.append("Agotamiento de vendedores FUERTE confirmado")
    elif agot_nivel == 2:
        score += 15
        razones.append("Agotamiento de vendedores MODERADO confirmado")
    elif agot_nivel == 1:
        score += 8
        razones.append("Primeras señales de agotamiento detectadas")
    else:
        alertas.append("Agotamiento de vendedores sin confirmar")

    if len(velas_rebote) > 0:
        score += 15
        razones.append(f"Patron de vela: {velas_rebote[-1][1]} detectado")

    if precio_vs_bb_low < 0:
        score += 15
        razones.append("Precio bajo BB inferior — zona de reversion estadistica")
    elif precio_vs_bb_low < 2:
        score += 5
        razones.append("Precio rozando BB inferior")

    return max(0, min(100, score)), razones, alertas

# ─────────────────────────────────────────────
# MOTOR HMM
# ─────────────────────────────────────────────
class MaquinaDineroLino:
    def __init__(self):
        self.modelo = hmm.GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=2000, random_state=42, tol=1e-5
        )

    def analizar(self, ticker):
        try:
            t  = yf.Ticker(ticker)
            df = descargar_datos(ticker, "2y", "1d")
            if df.empty or len(df) < 60:
                return None, "No hay suficientes datos historicos"

            close       = df["Close"].squeeze()
            retornos    = np.log(close / close.shift(1)).dropna()
            volatilidad = retornos.rolling(5).std().dropna()
            momentum    = close.pct_change(5).dropna()
            min_len     = min(len(retornos), len(volatilidad), len(momentum))
            X = np.column_stack([
                retornos.values[-min_len:],
                volatilidad.values[-min_len:],
                momentum.values[-min_len:]
            ])
            self.modelo.fit(X)
            gc.collect()  # liberar memoria post-HMM
            _, states     = self.modelo.decode(X, algorithm="viterbi")
            means         = self.modelo.means_[:, 0]
            bull_state    = int(np.argmax(means))
            bear_state    = int(np.argmin(means))
            estado_actual = int(states[-1])
            if estado_actual == bull_state:
                estado_hmm = "ALCISTA"
            elif estado_actual == bear_state:
                estado_hmm = "BAJISTA"
            else:
                estado_hmm = "LATERAL"

            rsi_serie        = calcular_rsi(close)
            rsi              = rsi_serie.iloc[-1]
            stoch_k, stoch_d = calcular_stoch_rsi(close)
            sk, sd           = stoch_k.iloc[-1], stoch_d.iloc[-1]
            macd, signal, _  = calcular_macd(close)
            macd_val         = macd.iloc[-1]
            signal_val       = signal.iloc[-1]

            df_chart = descargar_datos(ticker, "3mo", "1d")

            agot_ok, agot_nivel, agot_desc, agot_detalles = detectar_agotamiento_vendedores(df_chart)
            velas_rev   = detectar_vela_rebote(df_chart.tail(10))

            _, _, bb_low_s = calcular_bollinger(close)
            bb_low_val     = bb_low_s.iloc[-1]

            precio_rt, cambio_rt = obtener_precio_realtime(ticker)
            if precio_rt:
                precio_actual = precio_rt
                cambio        = cambio_rt
            else:
                precio_actual = float(close.iloc[-1])
                cambio = ((precio_actual - float(close.iloc[-2])) / float(close.iloc[-2])) * 100

            precio_vs_bb = ((precio_actual - bb_low_val) / bb_low_val) * 100

            score, razones, alertas = calcular_score_rebote(
                estado_hmm, rsi, sk, sd,
                macd_val, signal_val,
                agot_nivel, velas_rev, precio_vs_bb
            )

            niveles = calcular_niveles_trading(df_chart, precio_actual)

            if score >= 70:
                senal = "ENTRAR — REBOTE PROBABLE"
                color = "#00ff88"
            elif score >= 55:
                senal = "PREPARARSE — REBOTE POSIBLE"
                color = "#FFD700"
            elif score <= 30:
                senal = "NO ENTRAR — TENDENCIA BAJISTA"
                color = "#ff4444"
            else:
                senal = "NEUTRO — SIN SENAL CLARA"
                color = "#FF8C00"

            senales_bt = []
            for i in range(30, len(close)):
                r            = calcular_rsi(close.iloc[:i+1]).iloc[-1]
                sk_bt, sd_bt = calcular_stoch_rsi(close.iloc[:i+1])
                sk_v         = sk_bt.iloc[-1]
                sd_v         = sd_bt.iloc[-1]
                m, s, _      = calcular_macd(close.iloc[:i+1])
                mv, sv       = m.iloc[-1], s.iloc[-1]
                sc, _, _     = calcular_score_rebote(
                    estado_hmm, r, sk_v, sd_v, mv, sv, 0, [], 0
                )
                senales_bt.append(sc)

            senales_series = pd.Series(senales_bt)
            compras = (senales_series >= 70).sum()
            ventas  = (senales_series <= 30).sum()
            total   = len(senales_series)

            return {
                "senal": senal, "score": score, "color": color,
                "estado_hmm": estado_hmm,
                "precio": precio_actual, "cambio": cambio,
                "rsi": round(rsi, 1),
                "stoch_k": round(sk, 1), "stoch_d": round(sd, 1),
                "macd": round(macd_val, 4), "signal_line": round(signal_val, 4),
                "agot_ok": agot_ok, "agot_nivel": agot_nivel,
                "agot_desc": agot_desc, "agot_detalles": agot_detalles,
                "velas_rev": velas_rev,
                "razones": razones, "alertas": alertas,
                "niveles": niveles,
                "compras_bt": compras, "ventas_bt": ventas, "total_bt": total,
                "close": close, "df_chart": df_chart, "datos": len(df)
            }, None

        except Exception as e:
            return None, str(e)

# ─────────────────────────────────────────────
# GRAFICA PROFESIONAL (SWING)
# ─────────────────────────────────────────────
def grafica_rebote_profesional(df, ticker, nombre, niveles, velas_rev):
    if df.empty or len(df) < 10:
        return None

    close              = df['Close'].squeeze()
    ema9               = close.ewm(span=9).mean()
    ema21              = close.ewm(span=21).mean()
    stoch_k, stoch_d   = calcular_stoch_rsi(close)
    macd, signal, hist = calcular_macd(close)
    bb_up, _, bb_low   = calcular_bollinger(close)

    colores_velas = ['#00ff88' if c >= o else '#ff4444'
                     for c, o in zip(df['Close'], df['Open'])]
    colores_hist  = ['#00ff88' if v >= 0 else '#ff4444' for v in hist]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.48, 0.14, 0.19, 0.19],
        subplot_titles=(
            f"Velas Diarias — {nombre} ({ticker})",
            "Volumen",
            "Stoch RSI — Senal de Rebote",
            "MACD"
        )
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name="Precio",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88',  decreasing_fillcolor='#ff4444',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=bb_up, name="BB Superior",
        line=dict(color='#555577', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=bb_low, name="BB Inferior",
        line=dict(color='#335577', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(0,100,255,0.05)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema9,  name="EMA 9",
        line=dict(color='#FFD700', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name="EMA 21",
        line=dict(color='#00BFFF', width=1.5)), row=1, col=1)

    primer_idx = df.index[max(0, len(df)-30)]
    ultimo_idx = df.index[-1]
    for y_val, color, etiqueta in [
        (niveles['entrada'],    '#00ff88', f"Entrada ${niveles['entrada']:.2f}"),
        (niveles['stop_loss'],  '#ff4444', f"Stop ${niveles['stop_loss']:.2f}"),
        (niveles['objetivo_1'], '#FFD700', f"T1 ${niveles['objetivo_1']:.2f}"),
        (niveles['objetivo_2'], '#00BFFF', f"T2 ${niveles['objetivo_2']:.2f}"),
    ]:
        fig.add_shape(type="line",
            x0=primer_idx, x1=ultimo_idx, y0=y_val, y1=y_val,
            line=dict(color=color, width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=ultimo_idx, y=y_val, text=etiqueta,
            showarrow=False, xanchor="right",
            font=dict(color=color, size=10),
            bgcolor="rgba(0,0,0,0.5)", row=1, col=1)

    for fecha, tipo_vela, color_vela in velas_rev[-3:]:
        if fecha in df.index:
            precio_vela = float(df.loc[fecha, 'Low']) * 0.998
            fig.add_trace(go.Scatter(
                x=[fecha], y=[precio_vela],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=14, color=color_vela),
                text=[tipo_vela], textposition="bottom center",
                textfont=dict(size=9, color=color_vela),
                name=tipo_vela, showlegend=False
            ), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df['Volume'],
        name="Volumen", marker_color=colores_velas, opacity=0.7), row=2, col=1)
    vol_avg = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_avg, name="Vol MA20",
        line=dict(color='#888888', width=1, dash='dot')), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=stoch_k,
        name="Stoch K", line=dict(color='#00ff88', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=stoch_d,
        name="Stoch D", line=dict(color='#ff4444', width=1.5, dash='dot')), row=3, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ff4444", opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="#00ff88", opacity=0.08, line_width=0, row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ff4444", line_width=1, row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#00ff88", line_width=1, row=3, col=1)

    fig.add_trace(go.Bar(x=df.index, y=hist,
        name="Histograma", marker_color=colores_hist, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd,   name="MACD",
        line=dict(color='#00BFFF', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal",
        line=dict(color='#FF8C00', width=1.5)), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#888888", line_width=1, row=4, col=1)

    fig.update_layout(
        height=820,
        paper_bgcolor='#0f0c29', plot_bgcolor='#0f0c29',
        font=dict(color='#ffffff', size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    bgcolor='rgba(0,0,0,0.4)', font=dict(size=9)),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         showspikes=True, spikecolor="#00ff88",
                         spikethickness=1, spikemode="across", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor='#1a1a2e', zeroline=False,
                         tickfont=dict(size=9), row=i, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig

# ─────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────
if "ticker_sel" not in st.session_state: st.session_state.ticker_sel = ""
if "nombre_sel" not in st.session_state: st.session_state.nombre_sel = ""
if "pais_sel"   not in st.session_state: st.session_state.pais_sel   = ""

col_s, col_b = st.columns([4, 1])
with col_s:
    busqueda = st.text_input("Escribe empresa o ticker:",
        placeholder="Ej: Tesla, Apple, FEMSA, Cemex, Bitcoin...")
with col_b:
    st.write(""); st.write("")
    buscar_btn = st.button("ANALIZAR", use_container_width=True)

if busqueda and len(busqueda) >= 2 and not buscar_btn:
    with st.spinner("Buscando..."):
        sugs = buscar_sugerencias(busqueda)
    if sugs:
        opciones  = [s["label"] for s in sugs]
        seleccion = st.radio("Selecciona el activo:", opciones,
                             key="radio_sug", label_visibility="visible")
        idx = opciones.index(seleccion)
        st.session_state.ticker_sel = sugs[idx]["ticker"]
        st.session_state.nombre_sel = sugs[idx]["nombre"]
        st.session_state.pais_sel   = sugs[idx]["pais"]
        st.info(f"Seleccionado: **{st.session_state.nombre_sel}** "
                f"`{st.session_state.ticker_sel}` [{st.session_state.pais_sel}]")

col_r1, _ = st.columns([1, 5])
with col_r1:
    auto_refresh = st.checkbox("Auto-refresh 15 min")
if auto_refresh:
    import time; time.sleep(900); st.rerun()

ticker_a_usar = st.session_state.ticker_sel or busqueda.upper().strip()
nombre_a_usar = st.session_state.nombre_sel or busqueda

if buscar_btn and ticker_a_usar:
    with st.spinner("AI.Lino analizando..."):
        maquina          = MaquinaDineroLino()
        resultado, error = maquina.analizar(ticker_a_usar)

    if error:
        st.error(f"Error: {error}")
    elif resultado:

        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #0f0c29, #302b63); 
             padding: 30px; border-radius: 15px; text-align: center;
             border: 3px solid {resultado["color"]}; margin: 15px 0;'>
            <h1 style='color: {resultado["color"]}; font-size: 2.8em;'>{resultado["senal"]}</h1>
            <h2 style='color: white;'>Score AI.LINO: {resultado["score"]}/100</h2>
            <h3 style='color: #aaaaaa;'>Precio en tiempo real:
            <span style='color:white; font-weight:bold'> ${resultado["precio"]:.2f}</span>
            <span style='color:{"#00ff88" if resultado["cambio"]>0 else "#ff4444"}'>
            ({resultado["cambio"]:+.2f}%)</span></h3>
        </div>
        """, unsafe_allow_html=True)

        # Niveles swing
        niv = resultado["niveles"]
        st.markdown("### Niveles de Trading")
        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("Entrada Ideal",      f"${niv['entrada']:.2f}")
        n2.metric("Stop Loss",          f"${niv['stop_loss']:.2f}",
                  delta=f"-{niv['riesgo_pct']:.1f}%", delta_color="inverse")
        n3.metric("Objetivo 1",         f"${niv['objetivo_1']:.2f}")
        n4.metric("Objetivo 2",         f"${niv['objetivo_2']:.2f}",
                  delta=f"+{niv['pot_ganancia']:.1f}%")
        n5.metric("Riesgo / Beneficio", f"1 : {niv['rr_ratio']:.1f}")
        st.caption(f"ATR: ${niv['atr']:.2f} | Soporte: ${niv['soporte']:.2f} | "
                   f"Resistencia: ${niv['resistencia']:.2f} | BB Inf: ${niv['bb_low']:.2f}")

        # ══════════════════════════════════════════════════
        # MÓDULO PISO INTRADÍA — NUEVO
        # ══════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0a0a1a, #1a1a3e);
             padding: 12px 20px; border-radius: 10px; border-left: 4px solid #FF8C00;
             margin-bottom: 10px;'>
            <h3 style='color: #FF8C00; margin: 0;'>⚡ MÓDULO PISO INTRADÍA — Velas 5 minutos</h3>
            <p style='color: #888; margin: 4px 0 0 0; font-size: 0.85em;'>
            Detecta el piso del día en tiempo real · Ideal para operaciones de rango intradía
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Analizando momentum dominante..."):
            mom = detectar_momentum_dominante(ticker_a_usar)

        if "error" not in mom:
            # ── SEMÁFORO PRINCIPAL ──
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f0c29, #1a1a3e);
                 padding: 20px; border-radius: 12px; text-align: center;
                 border: 3px solid {mom["color_sem"]}; margin: 10px 0;'>
                <div style='display:flex; align-items:center; justify-content:center; gap:16px;'>
                    <div style='font-size:3em;'>{mom["icono"]}</div>
                    <div>
                        <div style='color:{mom["color_sem"]}; font-size:1.6em; font-weight:bold;
                             line-height:1.1;'>{mom["titulo"]}</div>
                        <div style='color:#aaa; font-size:0.9em; margin-top:4px;'>
                            {mom["descripcion"]}
                        </div>
                    </div>
                </div>
                <div style='margin-top:12px; background:rgba(0,0,0,0.3); border-radius:8px;
                     padding:8px; color:{mom["color_sem"]}; font-size:1em;'>
                    {mom["recomendacion"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── MÉTRICAS DEL SEMÁFORO ──
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("ADX",
                      f"{mom['adx']}",
                      "↑ subiendo" if mom.get("adx_subiendo") else "↓ bajando")
            m2.metric("+DI (Alcista)", f"{mom['pdi']}")
            m3.metric("-DI (Bajista)", f"{mom['mdi']}")
            m4.metric("Velas Verdes",
                      f"{mom['velas_verdes']}/20",
                      delta=f"{mom['velas_verdes']-10:+d}")
            m5.metric("Vol Alcista",  f"{mom['vol_verde_pct']}%")
            m6.metric("Vol Bajista",  f"{mom['vol_rojo_pct']}%")

            # ── BARRA ADX VISUAL ──
            adx_pct = min(mom['adx'], 60) / 60 * 100
            if mom['adx'] < 20:
                color_adx = "#888888"; label_adx = "Sin tendencia — ideal para rebotes"
            elif mom['adx'] < 25:
                color_adx = "#FFD700"; label_adx = "Tendencia débil"
            elif mom['adx'] < 40:
                color_adx = "#FF8C00"; label_adx = "Tendencia moderada"
            else:
                color_adx = "#ff4444"; label_adx = "Tendencia fuerte — no ir en contra"

            st.markdown(f"""
            <div style='margin: 8px 0 14px 0;'>
                <div style='display:flex; justify-content:space-between;
                     color:#888; font-size:0.78em; margin-bottom:3px;'>
                    <span>ADX: <b style='color:{color_adx}'>{mom['adx']}</b>
                    — {label_adx}</span>
                    <span>EMAs: {'🟢 Cascada alcista' if mom.get('cascada_alcista')
                              else '🔴 Cascada bajista' if mom.get('cascada_bajista')
                              else '🟡 Sin cascada'}</span>
                </div>
                <div style='background:#1a1a2e; border-radius:6px; height:12px;'>
                    <div style='background:{color_adx}; width:{adx_pct}%;
                         height:100%; border-radius:6px; opacity:0.8;'></div>
                </div>
                <div style='display:flex; justify-content:space-between;
                     color:#444; font-size:0.68em; margin-top:2px;'>
                    <span>0 — Lateral</span>
                    <span>25 — Tendencia</span>
                    <span>40+ — Fuerza dominante</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

        with st.spinner("Analizando intradía (velas 5min)..."):
            id_result = detectar_piso_intraday(ticker_a_usar)

        if "error" in id_result:
            st.warning(f"Datos intradía no disponibles: {id_result['error']}")
        else:
            # Señal principal intradía
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #0f0c29, #1a1a3e);
                 padding: 25px; border-radius: 12px; text-align: center;
                 border: 3px solid {id_result["color"]}; margin: 10px 0;'>
                <h2 style='color: {id_result["color"]}; font-size: 2em; margin: 0;'>
                    {id_result["señal"]}
                </h2>
                <p style='color: #aaa; margin: 8px 0 0 0;'>
                    Confianza intradía: 
                    <span style='color: {id_result["color"]}; font-size: 1.3em; font-weight: bold;'>
                        {id_result["confianza"]}%
                    </span>
                    &nbsp;|&nbsp; Puntos detectados: <b style='color:white'>{id_result["puntos"]}</b>/14
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── CONTEXTO HISTÓRICO — Panel nuevo ──
            st.markdown("#### 🗓️ Contexto Histórico (10 días)")

            pos_pct  = id_result.get("posicion_10d_pct", 50)
            ratio_c  = id_result.get("ratio_caida", 0)
            ratio_v  = id_result.get("ratio_velocidad", 1)
            caida_h  = id_result.get("caida_hoy", 0)
            caida_p  = id_result.get("caida_pct", 0)
            rpd      = id_result.get("rango_prom_diario", 0)
            h10      = id_result.get("high_10d", 0)
            l10      = id_result.get("low_10d", 0)
            sops     = id_result.get("soportes_reales", [])

            # Color según posición en rango
            if pos_pct <= 20:
                color_pos = "#00ff88"; label_pos = "🟢 PISO DEL RANGO"
            elif pos_pct <= 35:
                color_pos = "#7CFC00"; label_pos = "🟢 ZONA BAJA"
            elif pos_pct >= 80:
                color_pos = "#ff4444"; label_pos = "🔴 TECHO DEL RANGO"
            elif pos_pct >= 65:
                color_pos = "#FF8C00"; label_pos = "🟠 ZONA ALTA"
            else:
                color_pos = "#FFD700"; label_pos = "🟡 ZONA MEDIA"

            # Color según velocidad de caída
            if ratio_v > 2.0:
                color_vel = "#ff4444"; label_vel = "🚨 CAÍDA LIBRE"
            elif ratio_v < 0.5:
                color_vel = "#00ff88"; label_vel = "✅ DESACELERANDO"
            else:
                color_vel = "#FFD700"; label_vel = "⚠️ NORMAL"

            # Color según ratio de caída hoy
            if ratio_c > 1.5:
                color_caida = "#ff4444"; label_caida = "🚨 EXCESIVA"
            elif ratio_c > 1.0:
                color_caida = "#FF8C00"; label_caida = "⚠️ ELEVADA"
            elif ratio_c > 0.3:
                color_caida = "#00ff88"; label_caida = "✅ NORMAL"
            else:
                color_caida = "#888888"; label_caida = "➡️ MÍNIMA"

            st.markdown(f"""
            <div style='background:#0d0d1f; border:1px solid #333; border-radius:10px;
                 padding:14px; margin-bottom:10px;'>
                <div style='display:flex; gap:10px; flex-wrap:wrap;'>
                    <div style='flex:1; min-width:130px; background:#1a1a2e; border-radius:8px;
                         padding:10px; text-align:center;'>
                        <div style='color:#888; font-size:0.75em;'>POSICIÓN EN RANGO 10d</div>
                        <div style='color:{color_pos}; font-size:1.4em; font-weight:bold;'>{pos_pct:.0f}%</div>
                        <div style='color:{color_pos}; font-size:0.8em;'>{label_pos}</div>
                        <div style='color:#555; font-size:0.7em;'>Max:${h10:.2f} Min:${l10:.2f}</div>
                    </div>
                    <div style='flex:1; min-width:130px; background:#1a1a2e; border-radius:8px;
                         padding:10px; text-align:center;'>
                        <div style='color:#888; font-size:0.75em;'>CAÍDA HOY vs PROMEDIO</div>
                        <div style='color:{color_caida}; font-size:1.4em; font-weight:bold;'>
                            {ratio_c:.1f}x
                        </div>
                        <div style='color:{color_caida}; font-size:0.8em;'>{label_caida}</div>
                        <div style='color:#555; font-size:0.7em;'>
                            Bajó ${caida_h:.2f} ({caida_p:.1f}%) | Prom: ${rpd:.2f}
                        </div>
                    </div>
                    <div style='flex:1; min-width:130px; background:#1a1a2e; border-radius:8px;
                         padding:10px; text-align:center;'>
                        <div style='color:#888; font-size:0.75em;'>VELOCIDAD DE CAÍDA</div>
                        <div style='color:{color_vel}; font-size:1.4em; font-weight:bold;'>
                            {ratio_v:.1f}x
                        </div>
                        <div style='color:{color_vel}; font-size:0.8em;'>{label_vel}</div>
                        <div style='color:#555; font-size:0.7em;'>vs promedio últimas 3h</div>
                    </div>
                    <div style='flex:1; min-width:130px; background:#1a1a2e; border-radius:8px;
                         padding:10px; text-align:center;'>
                        <div style='color:#888; font-size:0.75em;'>SOPORTES HISTÓRICOS</div>
                        {''.join([f"<div style='color:#9B59B6; font-size:1em; font-weight:bold;'>${s:.2f}</div>" for s in sops[:3]]) if sops else "<div style='color:#555'>Sin datos</div>"}
                        <div style='color:#555; font-size:0.7em;'>niveles testados</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Barra visual de posición en rango
            st.markdown(f"""
            <div style='margin-bottom:12px;'>
                <div style='color:#888; font-size:0.8em; margin-bottom:4px;'>
                    Posición precio en rango de 10 días
                    &nbsp;|&nbsp; 🟢 Piso &nbsp;←&nbsp; Comprar aquí &nbsp;→&nbsp; 🔴 Techo
                </div>
                <div style='background:#1a1a2e; border-radius:6px; height:18px; position:relative;'>
                    <div style='background: linear-gradient(to right, #00ff88, #FFD700, #ff4444);
                         height:100%; border-radius:6px; opacity:0.3;'></div>
                    <div style='position:absolute; top:0; left:{min(max(pos_pct,2),98)}%;
                         transform:translateX(-50%); height:18px; width:3px;
                         background:{color_pos};'></div>
                </div>
                <div style='display:flex; justify-content:space-between;
                     color:#555; font-size:0.7em;'>
                    <span>Piso ${l10:.2f}</span>
                    <span>Precio actual ${id_result["precio_actual"]:.2f}</span>
                    <span>Techo ${h10:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Niveles intradía
            st.markdown("#### 📍 Niveles Intradía (ATR 5min)")
            i1, i2, i3, i4, i5 = st.columns(5)
            i1.metric("Entrada Ahora",   f"${id_result['entrada']:.2f}")
            i2.metric("Stop Loss",       f"${id_result['stop_loss']:.2f}",
                      delta=f"-{id_result['riesgo_pct']:.2f}%", delta_color="inverse")
            i3.metric("Objetivo 1",      f"${id_result['objetivo1']:.2f}",
                      delta=f"+{id_result['ganancia_pct']:.2f}%")
            i4.metric("Objetivo 2",      f"${id_result['objetivo2']:.2f}")
            i5.metric("R/B Intradía",    f"1 : {id_result['rr']:.1f}")

            st.caption(
                f"Mín hoy: ${id_result['min_hoy']:.2f} | "
                f"Máx hoy: ${id_result['max_hoy']:.2f} | "
                f"ATR 5min: ${id_result['atr_5m']:.3f} | "
                f"Stoch RSI 5m: {id_result['stoch_k_5m']} | "
                f"RSI 5m: {id_result['rsi_5m']}"
            )

            # Señales y alertas intradía
            col_si, col_ai = st.columns(2)
            with col_si:
                st.markdown("**✅ Señales Detectadas**")
                if id_result["señales"]:
                    for s in id_result["señales"]:
                        st.success(s)
                else:
                    st.info("Sin señales de piso activas")
            with col_ai:
                st.markdown("**⚠️ Alertas**")
                if id_result["alertas"]:
                    for a in id_result["alertas"]:
                        st.warning(a)
                else:
                    st.success("Sin alertas críticas")

            # Gráfica intradía
            st.markdown("#### 📈 Gráfica Intradía 5 minutos")
            fig_id = grafica_intraday(
                id_result.get("df_5m"),
                id_result.get("df_1h"),
                id_result,
                ticker_a_usar,
                nombre_a_usar
            )
            if fig_id:
                st.plotly_chart(fig_id, use_container_width=True)

        st.markdown("---")
        # ══════════════════════════════════════════════════
        # FIN MÓDULO PISO INTRADÍA
        # ══════════════════════════════════════════════════

        # Agotamiento de vendedores
        st.markdown("### Agotamiento de Vendedores")
        nivel_colores = {0: "#ff4444", 1: "#FF8C00", 2: "#FFD700", 3: "#00ff88"}
        nivel_iconos  = {0: "Sin confirmar", 1: "Debil", 2: "Moderado", 3: "FUERTE"}
        nv = resultado["agot_nivel"]
        color_agot = nivel_colores[nv]
        st.markdown(f"""
        <div style='background:#1a1a2e; border:2px solid {color_agot};
             border-radius:12px; padding:16px; margin-bottom:12px;'>
            <h3 style='color:{color_agot}; margin:0;'>
                {nivel_iconos[nv]} — {resultado["agot_desc"]}
            </h3>
        </div>
        """, unsafe_allow_html=True)

        if resultado["agot_detalles"]:
            for detalle in resultado["agot_detalles"]:
                st.success(f"✅ {detalle}")
        else:
            st.warning("Ninguna capa de agotamiento confirmada todavia")

        if resultado["velas_rev"]:
            velas_str = " | ".join([v[1] for v in resultado["velas_rev"][-3:]])
            st.success(f"Patrones de vela: {velas_str}")

        st.markdown("### Analisis de Rebote")
        col_r, col_a = st.columns(2)
        with col_r:
            st.markdown("**Senales a Favor**")
            for r in resultado["razones"]:
                st.success(r)
        with col_a:
            st.markdown("**Alertas**")
            for a in resultado["alertas"]:
                st.warning(a)

        st.markdown("### Indicadores")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("HMM Viterbi", resultado["estado_hmm"])
        c2.metric("RSI",         resultado["rsi"],
                  "Sobreventa" if resultado["rsi"] < 35 else
                  "Sobrecompra" if resultado["rsi"] > 70 else "Neutral")
        c3.metric("Stoch RSI K", resultado["stoch_k"],
                  "Zona rebote" if resultado["stoch_k"] < 20 else
                  "Sobrecompra" if resultado["stoch_k"] > 80 else "")
        c4.metric("MACD",        resultado["macd"],
                  f"Signal: {resultado['signal_line']}")
        c5.metric("Datos",       f"{resultado['datos']} dias")

        st.markdown("### Backtesting (2 anos)")
        b1, b2, b3 = st.columns(3)
        b1.metric("Senales Compra",    resultado["compras_bt"])
        b2.metric("Senales No Entrar", resultado["ventas_bt"])
        pct = round((resultado["compras_bt"] + resultado["ventas_bt"]) /
                    resultado["total_bt"] * 100, 1)
        b3.metric("Actividad", f"{pct}%")

        st.markdown("### Grafica con Niveles de Entrada y Salida")
        fig = grafica_rebote_profesional(
            resultado["df_chart"], ticker_a_usar,
            nombre_a_usar, niv, resultado["velas_rev"]
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.caption(f"AI.LINO Pro | HMM + Stoch RSI + Agotamiento 5 Capas + Piso Intradía | {ticker_a_usar}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555;'>AI.Lino Pro 2026 — Herramienta educativa. No es asesoria financiera.</p>",
    unsafe_allow_html=True
)
