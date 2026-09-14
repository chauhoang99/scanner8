from datetime import datetime, time, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Oanda Session Opening Range (OR) Analyzer", layout="wide"
)

st.title("🌐 Oanda Market Session Opening Range (OR) Dashboard")
st.markdown(
    "Analyze Opening Range sizes, session expansion correlations, and reversion tendencies separated by major global market sessions."
)

# ---------------------------------------------------------
# LOAD SECRETS SAFELY FROM STREAMLIT CLOUD
# ---------------------------------------------------------
try:
    secret_token = st.secrets.get("oanda_api_token", "")
    secret_account = st.secrets.get("oanda_account_id", "")
    secret_env = st.secrets.get("oanda_env", "Practice")
except Exception:
    secret_token, secret_account, secret_env = "", "", "Practice"

# ---------------------------------------------------------
# SIDEBAR CONFIGURATION
# ---------------------------------------------------------
st.sidebar.header("⚙️ Global Settings")

env_index = 0 if secret_env == "Practice" else 1
oanda_env = st.sidebar.selectbox("Environment", ["Practice", "Live"], index=env_index)

if secret_token:
    st.sidebar.success("🔒 Token loaded from Secrets")
    api_token = secret_token
else:
    api_token = st.sidebar.text_input("Oanda API Token", type="password", value="")

# Instrument Selection
instrument_options = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "AUD/USD": "AUD_USD",
    "USD/JPY": "USD_JPY",
    "XAU/USD (Gold)": "XAU_USD",
    "BTC/USD": "BTC_USD",
}
selected_label = st.sidebar.selectbox(
    "Select Instrument", list(instrument_options.keys())
)
oanda_inst = instrument_options[selected_label]


# Pip multiplier calculation helper
def get_pip_multiplier(ticker):
    ticker_upper = ticker.upper()
    if "JPY" in ticker_upper:
        return 100
    elif any(x in ticker_upper for x in ["BTC", "XAU", "BCO"]):
        return 1
    else:
        return 10000


# ---------------------------------------------------------
# DATA FETCHING FROM OANDA
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_m5_data(instrument, count=4500, token=None, env="Practice"):
    if not token:
        return None
    domain = (
        "api-fxtrade.oanda.com" if env == "Live" else "api-fxpractice.oanda.com"
    )
    url = f"https://{domain}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    params = {"price": "M", "granularity": "M5", "count": count}

    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            candles = data.get("candles", [])
            if not candles:
                return None
            rows = []
            for c in candles:
                if c.get("complete", True):
                    time_dt = pd.to_datetime(c["time"])
                    mid = c["mid"]
                    rows.append(
                        {
                            "Time": time_dt,
                            "Open": float(mid["o"]),
                            "High": float(mid["h"]),
                            "Low": float(mid["l"]),
                            "Close": float(mid["c"]),
                        }
                    )
            df = pd.DataFrame(rows)
            if not df.empty:
                df.set_index("Time", inplace=True)
            return df
    except Exception:
        return None
    return None


# ---------------------------------------------------------
# SESSION PROCESSING LOGIC
# ---------------------------------------------------------
def run_session_analysis(
    df, or_start_h, or_start_m, or_dur_mins, sess_end_h, mult
):
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
        df = df.index.tz_localize("UTC") if hasattr(df.index, 'tz_localize') else df
        # Safe timezone alignment
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")

    df["Date"] = df.index.date
    grouped = df.groupby("Date")

    analysis_results = []

    or_start_time = time(or_start_h, or_start_m)
    start_total_mins = or_start_h * 60 + or_start_m + or_dur_mins
    end_h = (start_total_mins // 60) % 24
    end_m = start_total_mins % 60
    or_end_time = time(end_h, end_m)
    session_end_t = time(sess_end_h, 0)

    for date_val, day_df in grouped:
        time_index = day_df.index.time
        or_mask = (time_index >= or_start_time) & (time_index < or_end_time)
        or_df = day_df[or_mask]

        if or_df.empty or len(or_df) < 2:
            continue

        or_high = or_df["High"].max()
        or_low = or_df["Low"].min()
        or_size = (or_high - or_low) * mult
        or_mid = (or_high + or_low) / 2

        if or_size <= 0:
            continue

        # Post-OR Session Window
        session_mask = (time_index >= or_end_time) & (time_index <= session_end_t)
        sess_df = day_df[session_mask]

        if sess_df.empty:
            continue

        session_high = sess_df["High"].max()
        session_low = sess_df["Low"].min()
        session_range = (
            max(session_high, or_high) - min(session_low, or_low)
        ) * mult

        # Max Extension from OR boundaries
        max_up_extension = max(0, sess_df["High"].max() - or_high) * mult
        max_down_extension = max(0, or_low - sess_df["Low"].min()) * mult
        max_extension = max(max_up_extension, max_down_extension)

        close_price = sess_df["Close"].iloc[-1]
        dist_from_mid_close = abs(close_price - or_mid) * mult

        analysis_results.append(
            {
                "Date": date_val,
                "OR_Size": or_size,
                "Session_Range": session_range,
                "Max_Extension": max_extension,
                "Dist_From_Mid_Close": dist_from_mid_close,
            }
        )

    return pd.DataFrame(analysis_results)


# ---------------------------------------------------------
# RENDER DASHBOARD TABS
# ---------------------------------------------------------
if not api_token:
    st.warning(
        "⚠️ Please provide your `oanda_api_token` in Streamlit Cloud Secrets or the sidebar."
    )
else:
    with st.spinner(f"Fetching M5 data for {selected_label}..."):
        raw_df = fetch_m5_data(oanda_inst, count=4500, token=api_token, env=oanda_env)

    if raw_df is None or raw_df.empty:
        st.error("Failed to fetch Oanda candles. Check your credentials or token.")
    else:
        mult = get_pip_multiplier(oanda_inst)
        unit_label = "pips" if mult in [100, 10000] else "pts"

        # Create Session Tabs
        tab_tokyo, tab_london, tab_ny = st.tabs(
            ["🇯🇵 Tokyo Session", "🇬🇧 London Session", "🇺🇸 New York Session"]
        )

        # =========================================================
        # TOKYO SESSION TAB
        # =========================================================
        with tab_tokyo:
            st.subheader("🇯🇵 Tokyo Session Analysis")
            col_s1, col_s2, col_s3 = st.columns(3)
            tokyo_or_h = col_s1.selectbox(
                "Tokyo OR Start Hour (UTC)", list(range(0, 24)), index=0, key="t_h"
            )
            tokyo_dur = col_s2.selectbox(
                "Tokyo OR Duration", [15, 30, 60, 120], index=2, key="t_d"
            )
            tokyo_end_h = col_s3.selectbox(
                "Tokyo Session End Hour (UTC)",
                list(range(0, 24)),
                index=8,
                key="t_e",
            )

            df_tokyo = run_session_analysis(
                raw_df, tokyo_or_h, 0, tokyo_dur, tokyo_end_h, mult
            )

            if df_tokyo.empty:
                st.info(
                    "No data matched the Tokyo session filters. Adjust hours or check timeframe."
                )
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    f"Avg OR Size ({unit_label})",
                    f"{df_tokyo['OR_Size'].mean():.1f}",
                )
                m2.metric(
                    f"Avg Session Range ({unit_label})",
                    f"{df_tokyo['Session_Range'].mean():.1f}",
                )
                corr_t = df_tokyo["OR_Size"].corr(df_tokyo["Session_Range"])
                m3.metric("OR Size vs Session Range Correlation", f"{corr_t:.2f}")

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    fig_t1 = px.scatter(
                        df_tokyo,
                        x="OR_Size",
                        y="Session_Range",
                        labels={
                            "OR_Size": f"Tokyo OR Size ({unit_label})",
                            "Session_Range": f"Tokyo Session Range ({unit_label})",
                        },
                        title="OR Size vs. Session Range (Tokyo)",
                    )
                    st.plotly_chart(fig_t1, use_container_width=True)
                with c2:
                    df_tokyo["OR_Category"] = np.where(
                        df_tokyo["OR_Size"] <= df_tokyo["OR_Size"].median(),
                        "Small OR",
                        "Large OR",
                    )
                    fig_t2 = px.box(
                        df_tokyo,
                        x="OR_Category",
                        y="Max_Extension",
                        labels={
                            "OR_Category": "Tokyo OR Classification",
                            "Max_Extension": f"Post-OR Max Extension ({unit_label})",
                        },
                        title="Extension vs Reversion (Tokyo)",
                    )
                    st.plotly_chart(fig_t2, use_container_width=True)

        # =========================================================
        # LONDON SESSION TAB
        # =========================================================
        with tab_london:
            st.subheader("🇬🇧 London Session Analysis")
            col_l1, col_l2, col_l3 = st.columns(3)
            london_or_h = col_l1.selectbox(
                "London OR Start Hour (UTC)", list(range(0, 24)), index=8, key="l_h"
            )
            london_dur = col_l2.selectbox(
                "London OR Duration", [15, 30, 60, 120], index=2, key="l_d"
            )
            london_end_h = col_l3.selectbox(
                "London Session End Hour (UTC)",
                list(range(0, 24)),
                index=16,
                key="l_e",
            )

            df_london = run_session_analysis(
                raw_df, london_or_h, 0, london_dur, london_end_h, mult
            )

            if df_london.empty:
                st.info(
                    "No data matched the London session filters. Adjust hours or check timeframe."
                )
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    f"Avg OR Size ({unit_label})",
                    f"{df_london['OR_Size'].mean():.1f}",
                )
                m2.metric(
                    f"Avg Session Range ({unit_label})",
                    f"{df_london['Session_Range'].mean():.1f}",
                )
                corr_l = df_london["OR_Size"].corr(df_london["Session_Range"])
                m3.metric("OR Size vs Session Range Correlation", f"{corr_l:.2f}")

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    fig_l1 = px.scatter(
                        df_london,
                        x="OR_Size",
                        y="Session_Range",
                        labels={
                            "OR_Size": f"London OR Size ({unit_label})",
                            "Session_Range": f"London Session Range ({unit_label})",
                        },
                        title="OR Size vs. Session Range (London)",
                    )
                    st.plotly_chart(fig_l1, use_container_width=True)
                with c2:
                    df_london["OR_Category"] = np.where(
                        df_london["OR_Size"] <= df_london["OR_Size"].median(),
                        "Small OR",
                        "Large OR",
                    )
                    fig_l2 = px.box(
                        df_london,
                        x="OR_Category",
                        y="Max_Extension",
                        labels={
                            "OR_Category": "London OR Classification",
                            "Max_Extension": f"Post-OR Max Extension ({unit_label})",
                        },
                        title="Extension vs Reversion (London)",
                    )
                    st.plotly_chart(fig_l2, use_container_width=True)

        # =========================================================
        # NEW YORK SESSION TAB
        # =========================================================
        with tab_ny:
            st.subheader("🇺🇸 New York Session Analysis")
            col_n1, col_n2, col_n3 = st.columns(3)
            ny_or_h = col_n1.selectbox(
                "New York OR Start Hour (UTC)",
                list(range(0, 24)),
                index=13,
                key="n_h",
            )
            ny_dur = col_n2.selectbox(
                "New York OR Duration", [15, 30, 60, 120], index=2, key="n_d"
            )
            ny_end_h = col_n3.selectbox(
                "New York Session End Hour (UTC)",
                list(range(0, 24)),
                index=21,
                key="n_e",
            )

            df_ny = run_session_analysis(
                raw_df, ny_or_h, 0, ny_dur, ny_end_h, mult
            )

            if df_ny.empty:
                st.info(
                    "No data matched the New York session filters. Adjust hours or check timeframe."
                )
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    f"Avg OR Size ({unit_label})", f"{df_ny['OR_Size'].mean():.1f}"
                )
                m2.metric(
                    f"Avg Session Range ({unit_label})",
                    f"{df_ny['Session_Range'].mean():.1f}",
                )
                corr_n = df_ny["OR_Size"].corr(df_ny["Session_Range"])
                m3.metric("OR Size vs Session Range Correlation", f"{corr_n:.2f}")

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    fig_n1 = px.scatter(
                        df_ny,
                        x="OR_Size",
                        y="Session_Range",
                        labels={
                            "OR_Size": f"New York OR Size ({unit_label})",
                            "Session_Range": f"New York Session Range ({unit_label})",
                        },
                        title="OR Size vs. Session Range (New York)",
                    )
                    st.plotly_chart(fig_n1, use_container_width=True)
                with c2:
                    df_ny["OR_Category"] = np.where(
                        df_ny["OR_Size"] <= df_ny["OR_Size"].median(),
                        "Small OR",
                        "Large OR",
                    )
                    fig_n2 = px.box(
                        df_ny,
                        x="OR_Category",
                        y="Max_Extension",
                        labels={
                            "OR_Category": "New York OR Classification",
                            "Max_Extension": f"Post-OR Max Extension ({unit_label})",
                        },
                        title="Extension vs Reversion (New York)",
                    )
                    st.plotly_chart(fig_n2, use_container_width=True)