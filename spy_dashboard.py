import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import datetime, math
import pandas as pd
import numpy as np
from fetch_data import fetch_spy_options_all_expirations

st.title("📈 SPY Options Dashboard (with Greeks, Liquidity, Vol & Straddle)")

# ── 1) fetch ────────────────────────────────────────────────────────────────
df = fetch_spy_options_all_expirations()
if df.empty:
    st.error("No options data loaded. Check network or ticker symbol.")
    st.stop()

# ── 2) expiration dropdown ──────────────────────────────────────────────────
exps    = sorted(df["Expiration"].unique())
sel_exp = st.selectbox("Select expiration date", exps)
df      = df[df["Expiration"] == sel_exp]

# ── 3) strike slider ────────────────────────────────────────────────────────
min_s, max_s = int(df["strike"].min()), int(df["strike"].max())
low, high   = st.slider("Strike price range", min_s, max_s, (min_s, max_s))
df = df[(df["strike"] >= low) & (df["strike"] <= high)]

# ── 4) calculate Greeks ─────────────────────────────────────────────────────
underlying_price = yf.Ticker("SPY").history(period="1d")["Close"].iloc[-1]
today = datetime.date.today()

def norm_pdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)
def norm_cdf(x): return (1+math.erf(x/math.sqrt(2)))/2

def calculate_greeks(row):
    S, K = underlying_price, row["strike"]
    sigma = row["impliedVolatility"]
    exp_dt = datetime.datetime.strptime(row["Expiration"], "%Y-%m-%d").date()
    T = max((exp_dt - today).days/365, 1/365)
    r = 0.0
    d1 = (math.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    pdf, cdf1, cdf2 = norm_pdf(d1), norm_cdf(d1), norm_cdf(d2)
    delta =  cdf1 if row["Type"]=="Call" else (cdf1-1)
    gamma =  pdf/(S*sigma*math.sqrt(T))
    vega  =  S*pdf*math.sqrt(T)/100
    theta_c = -(S*pdf*sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*cdf2
    theta_p = -(S*pdf*sigma)/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*(1-cdf2)
    theta = theta_c if row["Type"]=="Call" else theta_p
    rho_c =  K*T*math.exp(-r*T)*cdf2
    rho_p = -K*T*math.exp(-r*T)*(1-cdf2)
    rho = rho_c if row["Type"]=="Call" else rho_p
    return pd.Series({"Delta":delta,"Gamma":gamma,"Vega":vega,"Theta":theta,"Rho":rho})

greeks_df = df.apply(calculate_greeks, axis=1)
df = pd.concat([df.reset_index(drop=True), greeks_df], axis=1)

# ── 5) split calls vs puts & show table ─────────────────────────────────────
calls = df[df["Type"]=="Call"]
puts  = df[df["Type"]=="Put"]

st.subheader(f"Options Chain with Greeks ({sel_exp})")
st.dataframe(df)

# ── 6) Top Liquid Strikes ───────────────────────────────────────────────────
st.subheader("🔍 Top 5 Strikes by Open Interest")
st.table(df.sort_values("openInterest", ascending=False).head(5)[["strike","Type","openInterest","volume"]])

st.subheader("🔍 Top 5 Strikes by Volume")
st.table(df.sort_values("volume", ascending=False).head(5)[["strike","Type","openInterest","volume"]])

# ── 7) IV vs Strike ─────────────────────────────────────────────────────────
st.subheader("Implied Volatility vs Strike")
fig1, ax1 = plt.subplots()
ax1.plot(calls["strike"], calls["impliedVolatility"]*100, label="Calls")
ax1.plot(puts["strike"],  puts["impliedVolatility"]*100, label="Puts")
ax1.set_xlabel("Strike"); ax1.set_ylabel("Implied Volatility (%)")
ax1.legend(); st.pyplot(fig1)

# ── 8) Premium vs Strike ────────────────────────────────────────────────────
st.subheader("Option Premium vs Strike")
calls_p = (calls["bid"]+calls["ask"])/2
puts_p  = (puts["bid"] +puts["ask"]) /2
fig2, ax2 = plt.subplots()
ax2.plot(calls["strike"], calls_p, label="Calls")
ax2.plot(puts["strike"],  puts_p,  label="Puts")
ax2.set_xlabel("Strike"); ax2.set_ylabel("Mid Premium")
ax2.legend(); st.pyplot(fig2)

# ── 9) SPY Spot & Vol Comparison ─────────────────────────────────────────────
st.subheader("SPY Spot & Realized vs Implied Volatility")
hist = yf.Ticker("SPY").history(period="60d")["Close"]
rets = hist.pct_change().dropna()
realized_vol = rets.rolling(20).std()*np.sqrt(252)*100
avg_imp_vol  = df["impliedVolatility"].mean()*100

fig3,(ax3,ax4)=plt.subplots(2,1,figsize=(8,6),sharex=True)
ax3.plot(hist.index, hist.values); ax3.set_ylabel("SPY Price")
ax4.plot(realized_vol.index, realized_vol.values, label="Realized Vol (20d)")
ax4.axhline(avg_imp_vol, color="orange", linestyle="--", label=f"Avg Imp Vol ({sel_exp})")
ax4.set_ylabel("Volatility (%)"); ax4.legend()
st.pyplot(fig3)

# ── 10) Straddle P/L Payoff ──────────────────────────────────────────────────
st.subheader("🔧 Straddle P/L at Expiration")

# Sidebar inputs
st.sidebar.header("Straddle Builder")
strike_list = sorted(df["strike"].unique())
chosen_strike = st.sidebar.selectbox("Pick strike", strike_list, index=strike_list.index(int(underlying_price)))
n_contracts   = st.sidebar.number_input("Contracts", min_value=1, value=1, step=1)

# cost = mid-premium of call + put at that strike
call_row = calls[calls["strike"]==chosen_strike].iloc[0]
put_row  = puts[puts["strike"]==chosen_strike].iloc[0]
cost_per = ((call_row["bid"]+call_row["ask"])/2 + (put_row["bid"]+put_row["ask"])/2)
total_cost = cost_per * n_contracts * 100

# break-evens
be_lower = chosen_strike - cost_per
be_upper = chosen_strike + cost_per

st.markdown(f"- **Cost per straddle (1 contract)**: ${cost_per:.2f} × 100 = ${cost_per*100:.0f}")
st.markdown(f"- **Total cost**: ${total_cost:,.0f}")
st.markdown(f"- **Break‑even points**: ${be_lower:.2f} & ${be_upper:.2f}")

# payoff curve
price_grid = np.linspace(chosen_strike*0.7, chosen_strike*1.3, 200)
payoff = (np.maximum(price_grid-chosen_strike,0)+np.maximum(chosen_strike-price_grid,0) - cost_per) \
         * n_contracts * 100

fig4, ax5 = plt.subplots()
ax5.plot(price_grid, payoff)
ax5.axhline(0, color="black", linewidth=0.5)
ax5.set_xlabel("SPY Price at Expiration")
ax5.set_ylabel("P/L ($)")
ax5.set_title(f"Straddle P/L for strike {chosen_strike}")
st.pyplot(fig4)