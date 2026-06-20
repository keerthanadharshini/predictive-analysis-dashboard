import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Sales Predictive Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


DARK_BG    = "#0D1117"
CARD_BG    = "#161B22"
BORDER     = "#21262D"
ACCENT     = "#58A6FF"     
ACCENT2    = "#3FB950"      
WARN       = "#F78166"      
TEXT_PRI   = "#E6EDF3"
TEXT_SEC   = "#8B949E"
FONT_MONO  = "'JetBrains Mono', 'Fira Code', monospace"


st.markdown(f"""
<style>
  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background: {DARK_BG};
      color: {TEXT_PRI};
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background: {CARD_BG};
      border-right: 1px solid {BORDER};
  }}
  [data-testid="stSidebar"] * {{ color: {TEXT_PRI} !important; }}

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── KPI cards ── */
  .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin: 0 0 28px 0;
  }}
  .kpi-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 20px 22px 18px;
      position: relative;
      overflow: hidden;
  }}
  .kpi-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
  }}
  .kpi-card.blue::before  {{ background: {ACCENT}; }}
  .kpi-card.green::before {{ background: {ACCENT2}; }}
  .kpi-card.coral::before {{ background: {WARN}; }}
  .kpi-card.gold::before  {{ background: #D29922; }}

  .kpi-label {{
      font-size: 11px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: {TEXT_SEC};
      margin-bottom: 8px;
  }}
  .kpi-value {{
      font-size: 28px;
      font-weight: 700;
      color: {TEXT_PRI};
      line-height: 1;
      font-family: {FONT_MONO};
  }}
  .kpi-delta {{
      font-size: 12px;
      margin-top: 6px;
      color: {TEXT_SEC};
  }}
  .kpi-delta .up   {{ color: {ACCENT2}; }}
  .kpi-delta .down {{ color: {WARN}; }}

  /* ── Section headers ── */
  .section-header {{
      font-size: 13px;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: {TEXT_SEC};
      border-bottom: 1px solid {BORDER};
      padding-bottom: 8px;
      margin: 32px 0 16px;
  }}

  /* ── Insight cards ── */
  .insight-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-top: 16px;
  }}
  .insight-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-left: 3px solid {ACCENT};
      border-radius: 8px;
      padding: 16px 18px;
  }}
  .insight-card.green {{ border-left-color: {ACCENT2}; }}
  .insight-card.coral {{ border-left-color: {WARN}; }}
  .insight-card .ins-title {{
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: {TEXT_SEC};
      margin-bottom: 6px;
  }}
  .insight-card .ins-body {{
      font-size: 14px;
      color: {TEXT_PRI};
      line-height: 1.55;
  }}

  /* ── Chart containers ── */
  .chart-wrapper {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 4px;
      margin-bottom: 20px;
  }}

  /* ── Page title ── */
  .page-title {{
      font-size: 26px;
      font-weight: 800;
      color: {TEXT_PRI};
      letter-spacing: -.01em;
  }}
  .page-sub {{
      font-size: 14px;
      color: {TEXT_SEC};
      margin-top: 2px;
      margin-bottom: 24px;
  }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_clean(path: str = "sales_history.csv") -> pd.DataFrame:
    """
    Load, clean, and feature-engineer the sales CSV.

    Cleaning steps
    --------------
    1. Parse dates, coerce errors → NaT
    2. Drop rows with null Date or Sales
    3. Remove Sales ≤ 0 (physically impossible)
    4. Remove statistical outliers (IQR × 3)
    5. Sort chronologically, reset index
    """
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df.dropna(subset=["Date", "Sales"], inplace=True)

    df = df[df["Sales"] > 0].copy()

    q1, q3 = df["Sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df = df[(df["Sales"] >= q1 - 3 * iqr) & (df["Sales"] <= q3 + 3 * iqr)]

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Month"]        = df["Date"].dt.month
    df["Year"]         = df["Date"].dt.year
    df["Month_Index"]  = np.arange(len(df))          
    df["YearMonth"]    = df["Date"].dt.to_period("M").astype(str)

    return df


@st.cache_data
def build_model(df: pd.DataFrame):
    """
    Train a polynomial (degree-2) Linear Regression on Month_Index.
    Returns: pipeline, predictions, r2, forecast_df
    """
    X = df[["Month_Index"]].values
    y = df["Sales"].values

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr",   LinearRegression()),
    ])
    pipe.fit(X, y)

    y_pred = pipe.predict(X)
    r2     = r2_score(y, y_pred)

    last_idx  = df["Month_Index"].max()
    last_date = df["Date"].max()
    future_indices = np.arange(last_idx + 1, last_idx + 7).reshape(-1, 1)
    future_dates   = [last_date + pd.DateOffset(months=i) for i in range(1, 7)]
    future_sales   = pipe.predict(future_indices)
    future_sales   = np.clip(future_sales, 0, None)

    forecast_df = pd.DataFrame({
        "Date":         future_dates,
        "Sales":        np.round(future_sales, 2),
        "Month_Index":  future_indices.flatten(),
        "Type":         "Forecast",
        "YearMonth":    [d.strftime("%Y-%m") for d in future_dates],
    })

    return pipe, y_pred, r2, forecast_df

try:
    df_raw = load_and_clean("sales_history.csv")
except FileNotFoundError:
    st.error("⚠️  **sales_history.csv not found.**  "
             "Run `python generate_data.py` first, then reload.")
    st.stop()

pipe, y_pred_insample, r2, forecast_df = build_model(df_raw)

with st.sidebar:
    st.markdown("## ⚙️  Filters")
    st.markdown("---")

    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    st.markdown("---")
    show_trend_line = st.checkbox("Show trend line on history chart", value=True)
    show_ci         = st.checkbox("Show confidence band on forecast", value=True)

    st.markdown("---")
    st.markdown(f"<span style='color:{TEXT_SEC};font-size:12px'>"
                f"Data: {len(df_raw)} monthly observations<br>"
                f"Model: Polynomial (degree 2) LR<br>"
                f"Forecast horizon: 6 months</span>",
                unsafe_allow_html=True)


df = df_raw[
    (df_raw["Date"].dt.date >= start_d) &
    (df_raw["Date"].dt.date <= end_d)
].copy()

X_filt   = df[["Month_Index"]].values
y_filt   = df["Sales"].values
y_fitted = pipe.predict(X_filt) if len(X_filt) > 0 else np.array([])

st.markdown(
    '<div class="page-title">📈 Sales Predictive Analytics</div>'
    '<div class="page-sub">Historical trend analysis · Linear regression forecasting · Business insights</div>',
    unsafe_allow_html=True,
)


total_sales    = df["Sales"].sum()
avg_monthly    = df["Sales"].mean()
next_month_fc  = forecast_df["Sales"].iloc[0] if len(forecast_df) > 0 else 0
mom_change     = ((df["Sales"].iloc[-1] - df["Sales"].iloc[-2]) / df["Sales"].iloc[-2] * 100
                  if len(df) >= 2 else 0)

def fmt(val: float) -> str:
    """Format large numbers with K / M suffix."""
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:,.0f}"

arrow_mom = "up" if mom_change >= 0 else "down"
sign_mom  = "+" if mom_change >= 0 else ""

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card blue">
    <div class="kpi-label">Total Historical Sales</div>
    <div class="kpi-value">{fmt(total_sales)}</div>
    <div class="kpi-delta">{len(df)} months in view</div>
  </div>
  <div class="kpi-card gold">
    <div class="kpi-label">Avg Monthly Sales</div>
    <div class="kpi-value">{fmt(avg_monthly)}</div>
    <div class="kpi-delta">Filtered period</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-label">Forecast — Next Month</div>
    <div class="kpi-value">{fmt(next_month_fc)}</div>
    <div class="kpi-delta"><span class="{arrow_mom}">{sign_mom}{mom_change:.1f}% MoM</span> last actual</div>
  </div>
  <div class="kpi-card coral">
    <div class="kpi-label">Model Accuracy (R²)</div>
    <div class="kpi-value">{r2*100:.1f}%</div>
    <div class="kpi-delta">Full dataset · poly LR</div>
  </div>
</div>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(color=TEXT_PRI, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    margin=dict(l=16, r=16, t=40, b=16),
    hovermode="x unified",
)

def apply_layout(fig, title=""):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=TEXT_SEC)),
                      **CHART_LAYOUT)
    return fig

st.markdown('<div class="section-header">Historical Sales Trend</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)

    fig_hist = go.Figure()

    # Area fill under the actual line
    fig_hist.add_trace(go.Scatter(
        x=df["Date"], y=df["Sales"],
        fill="tozeroy",
        fillcolor=f"rgba(88,166,255,0.08)",
        line=dict(color=ACCENT, width=2.5),
        mode="lines+markers",
        marker=dict(size=5, color=ACCENT),
        name="Actual Sales",
        hovertemplate="<b>%{x|%b %Y}</b><br>Sales: $%{y:,.0f}<extra></extra>",
    ))

    # Optional trend line overlay
    if show_trend_line and len(y_fitted) > 0:
        fig_hist.add_trace(go.Scatter(
            x=df["Date"], y=y_fitted,
            line=dict(color=WARN, width=1.8, dash="dot"),
            mode="lines",
            name="Trend (model fit)",
            hovertemplate="Trend: $%{y:,.0f}<extra></extra>",
        ))

    apply_layout(fig_hist, "Monthly Sales — Actual vs Trend")
    fig_hist.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">6-Month Sales Forecast</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)


    band_pct = 0.05
    fig_fc = go.Figure()

    if show_ci:
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
            y=pd.concat([forecast_df["Sales"] * (1 + band_pct),
                         forecast_df["Sales"][::-1] * (1 - band_pct)]),
            fill="toself",
            fillcolor=f"rgba(63,185,80,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="±5% band",
        ))

    fig_fc.add_trace(go.Bar(
        x=forecast_df["Date"],
        y=forecast_df["Sales"],
        marker_color=ACCENT2,
        marker_line_width=0,
        opacity=0.85,
        name="Forecast",
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
    ))

 
    for _, row in forecast_df.iterrows():
        fig_fc.add_annotation(
            x=row["Date"], y=row["Sales"],
            text=fmt(row["Sales"]),
            showarrow=False,
            yshift=10,
            font=dict(size=11, color=ACCENT2),
        )

    apply_layout(fig_fc, "Predicted Sales — Next 6 Months")
    fig_fc.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_fc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="section-header">Combined View — History + Forecast</div>',
            unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)

    fig_combo = go.Figure()

   
    fig_combo.add_trace(go.Scatter(
        x=df_raw["Date"], y=df_raw["Sales"],
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.07)",
        line=dict(color=ACCENT, width=2),
        mode="lines+markers",
        marker=dict(size=4, color=ACCENT),
        name="Historical Sales",
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: $%{y:,.0f}<extra></extra>",
    ))

    bridge_x = [df_raw["Date"].iloc[-1], forecast_df["Date"].iloc[0]]
    bridge_y = [df_raw["Sales"].iloc[-1], forecast_df["Sales"].iloc[0]]
    fig_combo.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        line=dict(color=TEXT_SEC, width=1.5, dash="dot"),
        mode="lines",
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_combo.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Sales"],
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.08)",
        line=dict(color=ACCENT2, width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6, color=ACCENT2, symbol="diamond"),
        name="Forecast",
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: $%{y:,.0f}<extra></extra>",
    ))

    fig_combo.add_vline(
        x=df_raw["Date"].iloc[-1],
        line_dash="dot",
        line_color=TEXT_SEC,
        annotation_text="  Forecast →",
        annotation_font=dict(color=TEXT_SEC, size=11),
        annotation_position="top left",
    )

    apply_layout(fig_combo, "Full Timeline: Historical Data + 6-Month Prediction")
    fig_combo.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_combo, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-header">Business Insights</div>', unsafe_allow_html=True)

best_month   = df_raw.loc[df_raw["Sales"].idxmax()]
worst_month  = df_raw.loc[df_raw["Sales"].idxmin()]
yoy_df       = df_raw.copy()
yoy_df["Year"] = yoy_df["Date"].dt.year
yearly       = yoy_df.groupby("Year")["Sales"].sum()
yoy_pct      = ((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100
                if len(yearly) >= 2 else 0)

last6_actual   = df_raw["Sales"].tail(6).sum()
next6_forecast = forecast_df["Sales"].sum()
fc_vs_actual   = (next6_forecast - last6_actual) / last6_actual * 100

df_raw["Quarter"] = df_raw["Date"].dt.quarter
qtr_avg   = df_raw.groupby("Quarter")["Sales"].mean()
peak_qtr  = f"Q{qtr_avg.idxmax()}"

recent_slope = (df_raw["Sales"].iloc[-1] - df_raw["Sales"].iloc[-3]) / 2
momentum     = "accelerating" if recent_slope > 0 else "decelerating"

total_months  = len(df_raw)
cagr_monthly  = (df_raw["Sales"].iloc[-1] / df_raw["Sales"].iloc[0]) ** (1 / total_months) - 1
cagr_annual   = (1 + cagr_monthly) ** 12 - 1

insights = [
    {
        "color": "blue",
        "title": " Peak Sales Month",
        "body": (f"<b>{best_month['Date'].strftime('%B %Y')}</b> was the strongest month on record "
                 f"at <b>{fmt(best_month['Sales'])}</b> — "
                 f"{((best_month['Sales']-avg_monthly)/avg_monthly*100):.0f}% above the period average. "
                 f"Strongest quarter overall: <b>{peak_qtr}</b>."),
    },
    {
        "color": "green",
        "title": "📈 Year-over-Year Growth",
        "body": (f"Sales grew <b>{yoy_pct:+.1f}%</b> in the most recent full year. "
                 f"Compounded monthly growth rate implies an annualised rate of <b>{cagr_annual*100:.1f}%</b>, "
                 f"suggesting a healthy upward trajectory."),
    },
    {
        "color": "coral",
        "title": "🔮 Forecast Outlook",
        "body": (f"The next 6 months are projected to total <b>{fmt(next6_forecast)}</b>, "
                 f"a <b>{fc_vs_actual:+.1f}%</b> shift versus the previous 6 months. "
                 f"Next-month target: <b>{fmt(next_month_fc)}</b>. "
                 f"Recent momentum is <b>{momentum}</b>."),
    },
    {
        "color": "blue",
        "title": "📉 Weakest Period",
        "body": (f"<b>{worst_month['Date'].strftime('%B %Y')}</b> recorded the lowest sales at "
                 f"<b>{fmt(worst_month['Sales'])}</b>. "
                 f"Identifying drivers of this dip can help build contingency plans for similar future periods."),
    },
    {
        "color": "green",
        "title": "🎯 Model Confidence",
        "body": (f"The polynomial regression model explains <b>{r2*100:.1f}%</b> of sales variance (R²). "
                 f"An R² above 90% indicates the trend captures underlying growth patterns well; "
                 f"residual variance reflects genuine market noise."),
    },
    {
        "color": "coral",
        "title": "💡 Recommended Action",
        "body": (f"Seasonality peaks in <b>{peak_qtr}</b> — allocate marketing budget 4–6 weeks before "
                 f"this window. Forecasted growth supports increasing inventory targets by "
                 f"<b>{max(fc_vs_actual, 5):.0f}%</b> for the upcoming two quarters."),
    },
]

cards_html = '<div class="insight-grid">'
for ins in insights:
    cards_html += f"""
    <div class="insight-card {ins['color']}">
      <div class="ins-title">{ins['title']}</div>
      <div class="ins-body">{ins['body']}</div>
    </div>"""
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)


st.markdown('<div class="section-header">Raw Data</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    with st.expander("📋 Historical Sales (filtered)", expanded=False):
        display_hist = df[["Date", "Sales"]].copy()
        display_hist["Date"]  = display_hist["Date"].dt.strftime("%Y-%m")
        display_hist["Sales"] = display_hist["Sales"].map("${:,.0f}".format)
        st.dataframe(display_hist, use_container_width=True, hide_index=True)

with col_b:
    with st.expander("🔮 6-Month Forecast", expanded=False):
        display_fc = forecast_df[["Date", "Sales"]].copy()
        display_fc["Date"]  = display_fc["Date"].dt.strftime("%Y-%m")
        display_fc["Sales"] = display_fc["Sales"].map("${:,.0f}".format)
        st.dataframe(display_fc, use_container_width=True, hide_index=True)

st.markdown(
    f"<div style='margin-top:48px;padding-top:16px;border-top:1px solid {BORDER};"
    f"color:{TEXT_SEC};font-size:11px;text-align:center'>"
    f"Sales Predictive Analytics · Polynomial Linear Regression · "
    f"Built with Streamlit, Pandas, Scikit-Learn, Plotly"
    f"</div>",
    unsafe_allow_html=True,
)
