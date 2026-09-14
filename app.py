from datetime import datetime, time, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Oanda Opening Range (OR) Correlation Analyzer", layout="wide"
)

st.title("📊 Oanda Opening Range (OR) Statistical & Correlation Analyzer")
st.markdown(
    "Analyze how opening range sizes correlate with session expansion, daily ranges, and price reversion tendencies."
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
st.sidebar.header("⚙️ Oanda & Analysis Settings")

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

lookback_days = st.sidebar.slider(
    "Historical Lookback (Days)", min_value=10, max_value=180, value=60, step=10
)

st.sidebar.markdown("---")
st.sidebar.subheader("⏰ Opening Range (OR) & Session Times (UTC)")

# Configure OR Time Range
or_start_hour = st.sidebar.slider("OR Start Hour (UTC)", 0, 23, 7)
or_start_minute = st.sidebar.selectbox("OR Start Minute", [0, 15, 30, 45], index=0)
or_duration_mins = st.sidebar.selectbox(
    "OR Duration (Minutes)", [15, 30, 60, 120], index=2
)

# Session Window for Tracking Extension/Reversion
session_end_hour = st.sidebar.slider(
    "Session End Hour (UTC)", or_start_hour, 23, 16
)

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
def fetch_m5_data(instrument, count=4000, token=None, env="Practice"):
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
        if response.status_code == 250 or response.status_code == 200:
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
# PROCESSING & ANALYSIS LOGIC
# ---------------------------------------------------------
def run_opening_range_analysis(
    df, or_start_h, or_start_m, or_dur, sess_end_h, mult
):
    if df is None or df.empty:
        return pd.DataFrame()

    # Ensure UTC timezone alignment
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")

    df["Date"] = df.index.date
    grouped = df.groupby("Date")

    analysis_results = []

    or_start_time = time(or_start_h, or_start_m)
    # Calculate OR end time
    start_total_mins = or_start_h * 60 + or_start_m + or_dur
    end_h = (start_total_mins // 60) % 24
    end_m = start_total_mins % 60
    or_end_time = time(end_h, end_m)
    session_end_t = time(sess_end_h, 0)

    for date_val, day_df in grouped:
        # Filter Opening Range Window
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

        # Filter Post-OR / Session Window (from OR end to session end)
        session_mask = (time_index >= or_end_time) & (time_index <= session_end_t)
        sess_df = day_df[session_mask]

        if sess_df.empty:
            continue

        session_high = sess_df["High"].max()
        session_low = sess_df["Low"].min()
        session_range = (
            max(session_high, or_high) - min(session_low, or_low)
        ) * mult

        # Daily full range
        daily_high = day_df["High"].max()
        daily_low = day_df["Low"].min()
        daily_range = (daily_high - daily_low) * mult

        # Post-OR Extension & Reversion Metrics
        # Max extension above OR High or below OR Low post-OR
        max_up_extension = (
            max(0, sess_df["High"].max() - or_high) * mult
        )
        max_down_extension = (
            max(0, or_low - sess_df["Low"].min()) * mult
        )
        max_extension = max(max_up_extension, max_down_extension)

        # Reversion metric: distance of session close from OR midpoint, or whether price reverted back inside OR after breaking out
        close_price = sess_df["Close"].iloc[-1]
        distance_from_mid_at_close = abs(close_price - or_mid) * mult

        analysis_results.append(
            {
                "Date": date_val,
                "OR_Size": or_size,
                "Session_Range": session_range,
                "Daily_Range": daily_range,
                "Max_Extension": max_extension,
                "Dist_From_Mid_Close": distance_from_mid_at_close,
                "Expanded_Far": max_extension > (1.5 * or_size),
            }
        )

    return pd.DataFrame(analysis_results)


# ---------------------------------------------------------
# DASHBOARD UI EXECUTION
# ---------------------------------------------------------
if not api_token:
    st.warning(
        "⚠️ Please add your `oanda_api_token` to your Streamlit Cloud Secrets or input it in the sidebar."
    )
else:
    with st.spinner(f"Fetching M5 data for {selected_label} from Oanda..."):
        raw_df = fetch_m5_data(
            oanda_inst, count=4500, token=api_token, env=oanda_env
        )

    if raw_df is None or raw_df.empty:
        st.error(
            "Failed to retrieve candle data from Oanda. Please verify your token and environment."
        )
    else:
        mult = get_pip_multiplier(oanda_inst)
        unit_label = "pips" if mult in [100, 10000] else "pts"

        df_stats = run_opening_range_analysis(
            raw_df,
            or_start_hour,
            or_start_minute,
            or_duration_mins,
            session_end_hour,
            mult,
        )

        if df_stats.empty:
            st.warning(
                "Not enough data matched the selected opening range and session filter criteria."
            )
        else:
            # Display Quick Summary Metrics
            st.subheader(
                f"📈 Statistical Breakdown ({len(df_stats)} Trading Days Analyzed)"
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                f"Avg OR Size ({unit_label})", f"{df_stats['OR_Size'].mean():.1f}"
            )
            col2.metric(
                f"Avg Session Range ({unit_label})",
                f"{df_stats['Session_Range'].mean():.1f}",
            )
            col3.metric(
                f"Avg Daily Range ({unit_label})",
                f"{df_stats['Daily_Range'].mean():.1f}",
            )
            col4.metric(
                "Avg Max Extension", f"{df_stats['Max_Extension'].mean():.1f}"
            )

            st.markdown("---")

            # --- PART 1: CORRELATION WITH SESSION & DAILY RANGE ---
            st.subheader("1️⃣ Correlation: Opening Range Size vs. Session & Daily Range")

            corr_session = df_stats["OR_Size"].corr(df_stats["Session_Range"])
            corr_daily = df_stats["OR_Size"].corr(df_stats["Daily_Range"])

            c1, c2 = st.columns(2)
            c1.info(
                f"**OR Size vs. Session Range Correlation:** `{corr_session:.2f}`"
            )
            c2.info(f"**OR Size vs. Daily Range Correlation:** `{corr_daily:.2f}`")

            fig_scatter1 = px.scatter(
                df_stats,
                x="OR_Size",
                y="Session_Range",
                labels={
                    "OR_Size": f"Opening Range Size ({unit_label})",
                    "Session_Range": f"Session Range ({unit_label})",
                },
                title="Opening Range Size vs. Session Max Range",
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)

            st.markdown("---")

            # --- PART 2: EXPANSION VS REVERSION TENDENCIES ---
            st.subheader(
                "2️⃣ Reversion vs. Extension Tendency Based on Opening Range Size"
            )
            st.markdown(
                "Do larger opening ranges run far away, or do they mean-revert? We categorize days by **Small OR** vs **Large OR** (relative to median) to check post-OR behavior."
            )

            median_or = df_stats["OR_Size"].median()
            df_stats["OR_Category"] = np.where(
                df_stats["OR_Size"] <= median_or, "Small OR", "Large OR"
            )

            grouped_cat = (
                df_stats.groupby("OR_Category")[
                    ["Max_Extension", "Dist_From_Mid_Close", "Session_Range"]
                ]
                .mean()
                .reset_index()
            )

            st.dataframe(
                grouped_cat.style.format(
                    {
                        "Max_Extension": "{:.1f}",
                        "Dist_From_Mid_Close": "{:.1f}",
                        "Session_Range": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

            fig_box = px.box(
                df_stats,
                x="OR_Category",
                y="Max_Extension",
                labels={
                    "OR_Category": "Opening Range Classification",
                    "Max_Extension": f"Post-OR Max Extension ({unit_label})",
                },
                title="Post-OR Extension Magnitude: Small vs. Large Opening Ranges",
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Reversion conclusion note
            st.markdown(
                "> **Key Takeaway Guidance:** If the correlation is close to 1.0, wider opening ranges reliably lead to expanded trending days. If max extensions flatten or drop proportionally for large ORs, price tends to exhibit exhaustion and reversion behavior."
            )