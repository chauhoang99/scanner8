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
    "Analyze Opening Range sizes, post-OR extension magnitudes, and reversion/breakout probabilities across global sessions."
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
st.sidebar.header("⚙️ Oanda & Settings")

env_index = 0 if secret_env == "Practice" else 1
oanda_env = st.sidebar.selectbox("Environment", ["Practice", "Live"], index=env_index)

if secret_token:
    st.sidebar.success("🔒 Token loaded from Secrets")
    api_token = secret_token
else:
    api_token = st.sidebar.text_input("Oanda API Token", type="password", value="")

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

lookback_days = st.sidebar.slider(
    "Historical Lookback (Days)", min_value=10, max_value=180, value=60, step=10
)


def get_pip_multiplier(ticker):
    ticker_upper = ticker.upper()
    if "JPY" in ticker_upper:
        return 100
    elif any(x in ticker_upper for x in ["BTC", "XAU", "BCO"]):
        return 1
    else:
        return 10000


# ---------------------------------------------------------
# PAGINATED OANDA DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def fetch_oanda_m5_paginated(
    instrument, days=60, token=None, env="Practice"
):
    if not token:
        return None
    domain = (
        "api-fxtrade.oanda.com" if env == "Live" else "api-fxpractice.oanda.com"
    )
    url = f"https://{domain}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    all_candles = []
    to_time = pd.Timestamp.now(tz="UTC")
    target_start = to_time - pd.Timedelta(days=days)

    while to_time > target_start:
        params = {
            "price": "M",
            "granularity": "M5",
            "count": 4000,
            "to": to_time.isoformat(),
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                break
            data = response.json()
            candles = data.get("candles", [])
            if not candles:
                break

            all_candles.extend(candles)
            earliest_time = pd.to_datetime(candles[0]["time"])
            if earliest_time >= to_time:
                break
            to_time = earliest_time

            if len(candles) < 10:
                break
        except Exception:
            break

    if not all_candles:
        return None

    rows = []
    for c in all_candles:
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
        df.drop_duplicates(subset=["Time"], inplace=True)
        df.set_index("Time", inplace=True)
        df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------
# SESSION PROCESSING LOGIC (WITH OUTCOME CLASSIFICATION)
# ---------------------------------------------------------
def run_session_analysis(
    df, or_start_h, or_start_m, or_dur_mins, sess_end_h, mult
):
    if df is None or df.empty:
        return pd.DataFrame()

    if df.index.tz is None:
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

        or_high = float(or_df["High"].max())
        or_low = float(or_df["Low"].min())
        or_size = (or_high - or_low) * mult
        or_mid = (or_high + or_low) / 2

        if or_size <= 0:
            continue

        session_mask = (time_index >= or_end_time) & (time_index <= session_end_t)
        sess_df = day_df[session_mask]

        if sess_df.empty:
            continue

        session_high = float(sess_df["High"].max())
        session_low = float(sess_df["Low"].min())
        session_range = (
            max(session_high, or_high) - min(session_low, or_low)
        ) * mult

        max_up_extension = max(0, session_high - or_high) * mult
        max_down_extension = max(0, or_low - session_low) * mult
        max_extension = max(max_up_extension, max_down_extension)

        close_price = float(sess_df["Close"].iloc[-1])
        dist_from_mid_close = abs(close_price - or_mid) * mult

        # Classification logic for Breakout Success vs Reversion:
        # Reversion = Session close ended up back inside the Opening Range boundaries
        # Sustained Breakout = Session close ended outside the Opening Range boundaries
        is_closed_inside = (close_price >= or_low) & (close_price <= or_high)
        outcome_label = (
            "Reversion (Closed Inside)"
            if is_closed_inside
            else "Sustained Breakout"
        )

        analysis_results.append(
            {
                "Date": date_val,
                "OR_Size": or_size,
                "Session_Range": session_range,
                "Max_Extension": max_extension,
                "Dist_From_Mid_Close": dist_from_mid_close,
                "Session_Outcome": outcome_label,
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
    with st.spinner(
        f"Fetching {lookback_days} days of M5 data for {selected_label} from Oanda..."
    ):
        raw_df = fetch_oanda_m5_paginated(
            oanda_inst, days=lookback_days, token=api_token, env=oanda_env
        )

    if raw_df is None or raw_df.empty:
        st.error(
            "Failed to fetch Oanda candles. Check your token or environment settings."
        )
    else:
        mult = get_pip_multiplier(oanda_inst)
        unit_label = "pips" if mult in [100, 10000] else "pts"

        tab_tokyo, tab_london, tab_ny = st.tabs(
            ["🇯🇵 Tokyo Session", "🇬🇧 London Session", "🇺🇸 New York Session"]
        )

        def render_session_tab(session_name, default_or_h, default_end_h):
            st.subheader(f"{session_name} Session Analysis")
            col_s1, col_s2, col_s3 = st.columns(3)
            or_h = col_s1.selectbox(
                "OR Start Hour (UTC)",
                list(range(0, 24)),
                index=default_or_h,
                key=f"{session_name}_h",
            )
            or_dur = col_s2.selectbox(
                "OR Duration (Mins)",
                [15, 30, 60, 120],
                index=2,
                key=f"{session_name}_d",
            )
            end_h = col_s3.selectbox(
                "Session End Hour (UTC)",
                list(range(0, 24)),
                index=default_end_h,
                key=f"{session_name}_e",
            )

            df_sess = run_session_analysis(
                raw_df, or_h, 0, or_dur, end_h, mult
            )

            if df_sess.empty:
                st.info("No data matched the session filters.")
                return

            m1, m2, m3 = st.columns(3)
            m1.metric(
                f"Avg OR Size ({unit_label})", f"{df_sess['OR_Size'].mean():.1f}"
            )
            m2.metric(
                f"Avg Session Range ({unit_label})",
                f"{df_sess['Session_Range'].mean():.1f}",
            )
            corr = df_sess["OR_Size"].corr(df_sess["Session_Range"])
            m3.metric("OR Size vs Session Range Correlation", f"{corr:.2f}")

            median_cutoff = df_sess["OR_Size"].median()
            st.caption(
                f"📊 **Benchmark Cutoff:** Median OR size is **{median_cutoff:.1f} {unit_label}**. Days below median are classified as 'Small OR'."
            )

            df_sess["OR_Category"] = np.where(
                df_sess["OR_Size"] <= median_cutoff, "Small OR", "Large OR"
            )

            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                # Chart 1: Magnitude / Extension
                fig_ext = px.box(
                    df_sess,
                    x="OR_Category",
                    y="Max_Extension",
                    labels={
                        "OR_Category": "OR Classification",
                        "Max_Extension": f"Post-OR Max Extension Magnitude ({unit_label})",
                    },
                    title=f"Post-OR Max Extension Magnitude ({session_name})",
                )
                st.plotly_chart(fig_ext, use_container_width=True)

            with c2:
                # Chart 2: Success Rate vs Reversion Rate (Normalized Percentage Bar Chart)
                outcome_counts = (
                    df_sess.groupby(["OR_Category", "Session_Outcome"])
                    .size()
                    .reset_index(name="Count")
                )
                total_per_cat = (
                    df_sess.groupby("OR_Category")
                    .size()
                    .reset_index(name="Total")
                )
                outcome_counts = pd.merge(outcome_counts, total_per_cat, on="OR_Category")
                outcome_counts["Percentage"] = (
                    outcome_counts["Count"] / outcome_counts["Total"]
                ) * 100

                fig_prob = px.bar(
                    outcome_counts,
                    x="OR_Category",
                    y="Percentage",
                    color="Session_Outcome",
                    barmode="group",
                    labels={
                        "OR_Category": "OR Classification",
                        "Percentage": "Probability / Frequency (%)",
                        "Session_Outcome": "Session Close Outcome",
                    },
                    title=f"Breakout Success vs. Reversion Rate ({session_name})",
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        with tab_tokyo:
            render_session_tab("Tokyo", 0, 8)
        with tab_london:
            render_session_tab("London", 8, 16)
        with tab_ny:
            render_session_tab("New York", 13, 21)