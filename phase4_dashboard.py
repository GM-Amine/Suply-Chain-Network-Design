"""
============================================================
SUPPLY CHAIN NETWORK DESIGN — OLIST DATASET
Phase 4: Streamlit Dashboard
============================================================
Run with: streamlit run phase4_dashboard.py
============================================================
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import pulp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Network Design — Brazil",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark industrial aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0f14;
    color: #e8e4dc;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #12151c;
    border-right: 1px solid #1f2433;
}
section[data-testid="stSidebar"] * { color: #e8e4dc !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #12151c;
    border: 1px solid #1f2433;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 0 20px rgba(255,140,50,0.04);
}
div[data-testid="metric-container"] label {
    color: #7a8099 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ff8c32 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

/* Headers */
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #e8e4dc !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #e8e4dc !important; }
h3 { font-family: 'Syne', sans-serif !important; font-weight: 600 !important; color: #c8c4bc !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #7a8099 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff8c32 !important;
    border-bottom: 2px solid #ff8c32 !important;
}

/* Sliders */
div[data-testid="stSlider"] * { color: #e8e4dc !important; }
.stSlider > div > div > div > div { background: #ff8c32 !important; }

/* Select boxes */
div[data-baseweb="select"] { background: #12151c !important; border-color: #1f2433 !important; }

/* DataFrames */
div[data-testid="stDataFrame"] { border: 1px solid #1f2433; border-radius: 6px; }

/* Dividers */
hr { border-color: #1f2433 !important; }

/* Plotly chart background override */
.js-plotly-plot { border-radius: 8px; overflow: hidden; }

/* Section labels */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #ff8c32;
    margin-bottom: 4px;
}
.kpi-row { display: flex; gap: 12px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — calibrated from Phase 3 results
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR            = "C:\M2 SPA\Supply Chain\outputs"
COST_PER_KM_PER_ORDER = 0.08
MAX_SERVICE_DISTANCE  = 3500

DC_FIXED_COSTS = {
    "SP": 150_000, "RJ": 130_000, "MG": 110_000, "RS": 100_000,
    "PR": 100_000, "SC":  90_000, "BA":  85_000, "GO":  80_000,
    "PE":  75_000, "CE":  75_000, "ES":  70_000, "MT":  65_000,
    "MS":  65_000, "MA":  60_000, "PA":  60_000, "DF":  80_000,
    "RN":  55_000, "PI":  50_000, "PB":  50_000, "AL":  50_000,
    "TO":  45_000, "SE":  45_000, "RO":  45_000, "AM":  55_000,
    "AP":  40_000, "AC":  40_000, "RR":  40_000,
}

STATE_NAMES = {
    "AC":"Acre","AL":"Alagoas","AP":"Amapá","AM":"Amazonas","BA":"Bahia",
    "CE":"Ceará","DF":"Distrito Federal","ES":"Espírito Santo","GO":"Goiás",
    "MA":"Maranhão","MT":"Mato Grosso","MS":"Mato Grosso do Sul",
    "MG":"Minas Gerais","PA":"Pará","PB":"Paraíba","PR":"Paraná",
    "PE":"Pernambuco","PI":"Piauí","RJ":"Rio de Janeiro",
    "RN":"Rio Grande do Norte","RS":"Rio Grande do Sul","RO":"Rondônia",
    "RR":"Roraima","SC":"Santa Catarina","SP":"São Paulo",
    "SE":"Sergipe","TO":"Tocantins",
}

# Plotly dark theme base
PLOTLY_DARK = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#12151c",
    font=dict(family="Syne, sans-serif", color="#e8e4dc"),
    xaxis=dict(gridcolor="#1f2433", linecolor="#1f2433"),
    yaxis=dict(gridcolor="#1f2433", linecolor="#1f2433"),
)
ACCENT   = "#ff8c32"
BLUE     = "#3b82f6"
GREEN    = "#22c55e"
MUTED    = "#7a8099"
DC_PALETTE = [ACCENT, BLUE, GREEN, "#a855f7", "#ec4899",
              "#14b8a6", "#f59e0b", "#ef4444"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    demand_df  = pd.read_csv(f"{OUTPUT_DIR}/demand_for_milp.csv")
    demand_df  = demand_df.dropna(subset=["centroid_lat","centroid_lng"])
    demand_df  = demand_df[demand_df["total_forecast"] > 0].reset_index(drop=True)
    dist_raw   = pd.read_csv(f"{OUTPUT_DIR}/distance_matrix_km.csv", index_col=0)
    monthly    = pd.read_csv(f"{OUTPUT_DIR}/monthly_demand.csv", parse_dates=["order_month"])
    sens_df    = pd.read_csv(f"{OUTPUT_DIR}/sensitivity_dc_count.csv")
    scenario_df= pd.read_csv(f"{OUTPUT_DIR}/scenario_results.csv")
    tornado_df = pd.read_csv(f"{OUTPUT_DIR}/tornado_sensitivity.csv")
    flows_df   = pd.read_csv(f"{OUTPUT_DIR}/baseline_flows.csv")
    return demand_df, dist_raw, monthly, sens_df, scenario_df, tornado_df, flows_df

demand_df, dist_raw, monthly, sens_df, scenario_df, tornado_df, flows_df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# MILP SOLVER (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def solve_milp(n_dcs: int, scenario: str):
    I = demand_df["demand_state"].tolist()
    J = [s for s in DC_FIXED_COSTS if s in dist_raw.index]

    col_map = {
        "Pessimistic (lower 95%)": "total_lower_95",
        "Baseline (LightGBM)"    : "total_forecast",
        "Optimistic (upper 95%)" : "total_upper_95",
    }
    dcol = col_map.get(scenario, "total_forecast")
    d    = dict(zip(demand_df["demand_state"], demand_df[dcol].clip(lower=0)))
    f    = {j: DC_FIXED_COSTS[j] for j in J}
    c    = {}
    for i in I:
        for j in J:
            if i in dist_raw.index and j in dist_raw.columns:
                c[(i,j)] = COST_PER_KM_PER_ORDER * float(dist_raw.loc[i,j])
            else:
                c[(i,j)] = 9999.0

    total_demand = sum(d.values())
    K = {j: total_demand * 8.0 / len(J) for j in J}
    forbidden = {(i,j) for i in I for j in J
                 if i in dist_raw.index and j in dist_raw.columns
                 and float(dist_raw.loc[i,j]) > MAX_SERVICE_DISTANCE}

    prob = pulp.LpProblem("CFLP", pulp.LpMinimize)
    y    = pulp.LpVariable.dicts("open", J, cat="Binary")
    x    = pulp.LpVariable.dicts("flow",
               [(i,j) for i in I for j in J], lowBound=0, cat="Continuous")

    prob += (pulp.lpSum(f[j]*y[j] for j in J) +
             pulp.lpSum(c[(i,j)]*x[(i,j)] for i in I for j in J
                        if (i,j) not in forbidden))

    for i in I:
        prob += pulp.lpSum(x[(i,j)] for j in J
                           if (i,j) not in forbidden) >= d[i]
    for j in J:
        prob += pulp.lpSum(x[(i,j)] for i in I
                           if (i,j) not in forbidden) <= K[j]*y[j]
    for i in I:
        for j in J:
            if (i,j) not in forbidden:
                prob += x[(i,j)] <= d[i]*y[j]
    for (i,j) in forbidden:
        prob += x[(i,j)] == 0
    prob += pulp.lpSum(y[j] for j in J) <= n_dcs

    pulp.PULP_CBC_CMD(timeLimit=60, msg=0, gapRel=0.01).solve(prob)

    open_dcs = [j for j in J if pulp.value(y[j]) > 0.5]
    flows    = {(i,j): pulp.value(x[(i,j)])
                for i in I for j in J
                if (i,j) not in forbidden
                and pulp.value(x[(i,j)]) is not None
                and pulp.value(x[(i,j)]) > 0.01}
    assignment = {}
    for i in I:
        candidates = [j for j in open_dcs if (i,j) not in forbidden]
        if candidates:
            assignment[i] = max(candidates, key=lambda j: flows.get((i,j), 0))

    fixed_cost     = sum(f[j] for j in open_dcs)
    transport_cost = sum(c[(i,j)]*v for (i,j),v in flows.items())
    total_orders   = sum(d.values())
    avg_dist       = sum(
        flows.get((i,j),0) * float(dist_raw.loc[i,j])
        for i in I for j in open_dcs
        if i in dist_raw.index and j in dist_raw.columns
        and (i,j) not in forbidden
    ) / max(total_orders, 1)

    util = {}
    for j in open_dcs:
        served  = sum(flows.get((i,j),0) for i in I)
        util[j] = served / K[j] * 100

    return dict(
        status         = pulp.LpStatus[prob.status],
        objective      = pulp.value(prob.objective),
        open_dcs       = open_dcs,
        flows          = flows,
        assignment     = assignment,
        fixed_cost     = fixed_cost,
        transport_cost = transport_cost,
        avg_dist_km    = avg_dist,
        utilization    = util,
        d              = d,
        f              = f,
        c              = c,
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Network Designer")
    st.markdown("---")

    st.markdown('<p class="section-label">Demand Scenario</p>', unsafe_allow_html=True)
    scenario = st.selectbox(
        "", ["Baseline (LightGBM)", "Pessimistic (lower 95%)", "Optimistic (upper 95%)"],
        label_visibility="collapsed"
    )

    st.markdown('<p class="section-label">Number of Distribution Centers</p>',
                unsafe_allow_html=True)
    n_dcs = st.slider("", min_value=1, max_value=8, value=6, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-label">Phase 3 Key Results</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:11px;line-height:2;color:#7a8099">
    Baseline optimal DCs : <span style="color:#ff8c32">6</span><br>
    Elbow point          : <span style="color:#ff8c32">4 DCs</span><br>
    Min total cost       : <span style="color:#ff8c32">R$ 1,011,446</span><br>
    Avg distance (base)  : <span style="color:#ff8c32">182 km</span><br>
    Total demand (3mo)   : <span style="color:#ff8c32">30,038 orders</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Top Demand States</p>', unsafe_allow_html=True)
    top5 = demand_df.nlargest(5, "total_forecast")[["demand_state","total_forecast","demand_share_%"]]
    for _, row in top5.iterrows():
        pct  = row["demand_share_%"]
        bar  = int(pct / 100 * 20)
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:11px;margin-bottom:6px">'
            f'<span style="color:#e8e4dc">{row["demand_state"]}</span> '
            f'<span style="color:#ff8c32">{"█"*bar}{"░"*(20-bar)}</span> '
            f'<span style="color:#7a8099">{pct:.1f}%</span></div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# SOLVE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Optimising network…"):
    result = solve_milp(n_dcs, scenario)

I         = demand_df["demand_state"].tolist()
d         = result["d"]
open_dcs  = result["open_dcs"]
assignment= result["assignment"]
flows     = result["flows"]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:2rem;margin-bottom:0">
  Supply Chain Network Design
  <span style="color:#ff8c32">— Brazil</span>
</h1>
<p style="color:#7a8099;font-family:'DM Mono',monospace;font-size:12px;margin-top:4px">
  Olist E-Commerce Dataset · MILP (CFLP) · LightGBM Demand Forecasts
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

baseline_cost = 1_011_446
delta_cost    = (result["objective"] - baseline_cost) / baseline_cost * 100 if result["objective"] else 0

k1.metric("Total Cost",       f"R$ {result['objective']:,.0f}",
          f"{delta_cost:+.1f}% vs baseline")
k2.metric("Fixed Cost",       f"R$ {result['fixed_cost']:,.0f}",
          f"{result['fixed_cost']/result['objective']*100:.1f}% of total")
k3.metric("Transport Cost",   f"R$ {result['transport_cost']:,.0f}",
          f"{result['transport_cost']/result['objective']*100:.1f}% of total")
k4.metric("DCs Opened",       f"{len(open_dcs)} / {n_dcs}",
          f"Avg util {np.mean(list(result['utilization'].values())):.0f}%"
          if result["utilization"] else "")
k5.metric("Avg Distance",     f"{result['avg_dist_km']:.0f} km",
          f"Scenario: {scenario.split('(')[0].strip()}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️  Network Map",
    "📊  Sensitivity",
    "🎯  Scenarios",
    "🌪️  Tornado",
    "📈  Demand Forecast",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NETWORK MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_map, col_table = st.columns([2, 1])

    with col_map:
        st.markdown("### Optimal Distribution Network")
        st.caption(f"Scenario: **{scenario}** · {n_dcs} DC(s) allowed · {len(open_dcs)} opened")

        coords = demand_df.set_index("demand_state")[
            ["centroid_lat","centroid_lng"]].to_dict("index")

        dc_color_map = {j: DC_PALETTE[k % len(DC_PALETTE)]
                        for k, j in enumerate(open_dcs)}

        m = folium.Map(location=[-14.2, -51.9], zoom_start=4,
                       tiles="CartoDB dark_matter")

        # Demand heatmap
        heat = [[coords[i]["centroid_lat"], coords[i]["centroid_lng"],
                 d.get(i,0)/max(d.values())]
                for i in I if i in coords]
        HeatMap(heat, radius=30, blur=20, min_opacity=0.2).add_to(m)

        # Flow lines
        for i in I:
            j = assignment.get(i)
            if not j or i not in coords or j not in coords or i == j:
                continue
            ci, cj   = coords[i], coords[j]
            flow_val = flows.get((i,j), 0)
            w        = 1 + flow_val / max(flows.values(), default=1) * 5
            folium.PolyLine(
                [[ci["centroid_lat"], ci["centroid_lng"]],
                 [cj["centroid_lat"], cj["centroid_lng"]]],
                weight=w, color=dc_color_map.get(j,"#888"),
                opacity=0.6,
                tooltip=f"{i}→{j}: {flow_val:.0f} orders",
            ).add_to(m)

        # Demand markers
        for i in I:
            if i not in coords:
                continue
            j   = assignment.get(i)
            col = dc_color_map.get(j, "#888")
            ci  = coords[i]
            folium.CircleMarker(
                [ci["centroid_lat"], ci["centroid_lng"]],
                radius=4 + d.get(i,0)/max(d.values())*14,
                color=col, fill=True, fill_opacity=0.65,
                popup=folium.Popup(
                    f"<b>{STATE_NAMES.get(i,i)} ({i})</b><br>"
                    f"Demand: {d.get(i,0):,.0f} orders<br>"
                    f"Served by DC: {j}",
                    max_width=180),
                tooltip=f"{i} → DC {j} | {d.get(i,0):,.0f} orders",
            ).add_to(m)

        # DC markers
        for j in open_dcs:
            if j not in coords:
                continue
            cj   = coords[j]
            util = result["utilization"].get(j,0)
            folium.Marker(
                [cj["centroid_lat"], cj["centroid_lng"]],
                icon=folium.Icon(
                    color="red" if util > 80 else "orange",
                    icon="industry", prefix="fa"),
                popup=folium.Popup(
                    f"<b>DC: {STATE_NAMES.get(j,j)} ({j})</b><br>"
                    f"Fixed cost: R${DC_FIXED_COSTS[j]:,}/mo<br>"
                    f"Utilization: {util:.1f}%",
                    max_width=200),
                tooltip=f"DC {j} | {util:.0f}% utilized",
            ).add_to(m)

        st_folium(m, width=700, height=500)

    with col_table:
        st.markdown("### DC Assignment")
        rows = []
        for i in sorted(I):
            j = assignment.get(i, "—")
            dist_val = (float(dist_raw.loc[i,j])
                        if j != "—" and i in dist_raw.index
                        and j in dist_raw.columns else 0)
            rows.append({
                "State"    : i,
                "Served by": j,
                "Orders"   : f"{d.get(i,0):,.0f}",
                "Dist (km)": f"{dist_val:.0f}",
            })
        assign_df = pd.DataFrame(rows)
        st.dataframe(assign_df, use_container_width=True, height=480,
                     hide_index=True)

        st.markdown("### DC Utilization")
        for j in sorted(open_dcs):
            util = result["utilization"].get(j,0)
            col  = dc_color_map.get(j, ACCENT)
            st.markdown(
                f'<div style="margin-bottom:6px;font-family:DM Mono,monospace;font-size:12px">'
                f'<span style="color:{col}">■</span> {j} '
                f'<span style="color:#7a8099">{util:.1f}%</span></div>',
                unsafe_allow_html=True)
            st.progress(min(util/100, 1.0))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Sensitivity Analysis — Number of DCs")

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Total Cost", "Cost Breakdown", "Avg Delivery Distance"],
    )

    # Total cost curve
    fig.add_trace(go.Scatter(
        x=sens_df["n_dcs"], y=sens_df["total_cost"]/1000,
        mode="lines+markers",
        line=dict(color=ACCENT, width=3),
        marker=dict(size=8, color=ACCENT),
        name="Total Cost",
        hovertemplate="<b>%{x} DCs</b><br>R$ %{y:,.0f}k<extra></extra>",
    ), row=1, col=1)

    # Highlight current selection
    if n_dcs in sens_df["n_dcs"].values:
        row_sel = sens_df[sens_df["n_dcs"] == n_dcs].iloc[0]
        fig.add_trace(go.Scatter(
            x=[row_sel["n_dcs"]], y=[row_sel["total_cost"]/1000],
            mode="markers",
            marker=dict(size=14, color=ACCENT, symbol="star",
                        line=dict(color="white", width=2)),
            name="Current", showlegend=True,
        ), row=1, col=1)

    # Stacked cost breakdown
    fig.add_trace(go.Bar(
        x=sens_df["n_dcs"], y=sens_df["fixed_cost"]/1000,
        name="Fixed", marker_color="#ef4444", opacity=0.85,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=sens_df["n_dcs"], y=sens_df["transport_cost"]/1000,
        name="Transport", marker_color=BLUE, opacity=0.85,
    ), row=1, col=2)

    # Distance curve
    fig.add_trace(go.Scatter(
        x=sens_df["n_dcs"], y=sens_df["avg_dist_km"],
        mode="lines+markers",
        line=dict(color=GREEN, width=3),
        marker=dict(size=8, color=GREEN),
        name="Avg Distance",
        hovertemplate="<b>%{x} DCs</b><br>%{y:.0f} km<extra></extra>",
    ), row=1, col=3)

    fig.update_layout(
        **PLOTLY_DARK,
        height=420,
        barmode="stack",
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(title_text="Number of DCs", gridcolor="#1f2433")
    fig.update_yaxes(gridcolor="#1f2433")
    fig.update_annotations(font=dict(color="#7a8099", size=12))

    st.plotly_chart(fig, use_container_width=True)

    # Marginal savings table
    st.markdown("### Marginal Cost Savings")
    display_sens = sens_df[["n_dcs","total_cost","fixed_cost",
                             "transport_cost","avg_dist_km"]].copy()
    display_sens.columns = ["DCs","Total Cost (R$)","Fixed (R$)",
                             "Transport (R$)","Avg Dist (km)"]
    for col in ["Total Cost (R$)","Fixed (R$)","Transport (R$)"]:
        display_sens[col] = display_sens[col].apply(lambda x: f"R$ {x:,.0f}")
    display_sens["Avg Dist (km)"] = display_sens["Avg Dist (km)"].apply(
        lambda x: f"{x:.0f} km")
    st.dataframe(display_sens, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Demand Scenario Comparison")
    st.caption("4 DCs (elbow point) — point forecasts from LightGBM, bounds from Prophet 95% CI")

    col_a, col_b = st.columns(2)

    with col_a:
        # Parse scenario costs from CSV
        try:
            s_df = scenario_df.copy()
            # Extract numeric total cost
            s_df["cost_num"] = s_df["Total Cost"].str.replace(r"[R$\s,]","",regex=True).astype(float)
            s_df["fixed_num"] = s_df["Fixed Cost"].str.replace(r"[R$\s,]","",regex=True).astype(float)
            s_df["trans_num"] = s_df["Transport Cost"].str.replace(r"[R$\s,]","",regex=True).astype(float)
            labels = [s.split("(")[0].strip() for s in s_df["Scenario"]]

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Bar(
                x=labels, y=s_df["fixed_num"]/1000,
                name="Fixed", marker_color="#ef4444", opacity=0.9,
            ))
            fig_sc.add_trace(go.Bar(
                x=labels, y=s_df["trans_num"]/1000,
                name="Transport", marker_color=BLUE, opacity=0.9,
            ))
            # Value labels
            for i, (fc, tc) in enumerate(zip(s_df["fixed_num"], s_df["trans_num"])):
                fig_sc.add_annotation(
                    x=labels[i], y=(fc+tc)/1000 + 10,
                    text=f"R${(fc+tc)/1000:.0f}k",
                    showarrow=False, font=dict(size=11, color="white"),
                )
            fig_sc.update_layout(
                **PLOTLY_DARK, barmode="stack", height=380,
                title="Total Cost by Scenario",
                yaxis_title="Cost (R$ thousands)",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=50, b=40),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not parse scenario CSV: {e}")

    with col_b:
        # Demand uncertainty band chart
        demand_plot = demand_df.sort_values("total_forecast", ascending=False).head(12)
        fig_unc = go.Figure()

        # CI band
        fig_unc.add_trace(go.Bar(
            name="Forecast Range",
            x=demand_plot["demand_state"],
            y=(demand_plot["total_upper_95"] - demand_plot["total_lower_95"]),
            base=demand_plot["total_lower_95"],
            marker_color=BLUE, opacity=0.25,
            hovertemplate="<b>%{x}</b><br>Lower: %{base:,.0f}<br>Upper: %{y:,.0f}<extra></extra>",
        ))
        # Point forecast
        fig_unc.add_trace(go.Scatter(
            x=demand_plot["demand_state"],
            y=demand_plot["total_forecast"],
            mode="markers+lines",
            marker=dict(color=ACCENT, size=9),
            line=dict(color=ACCENT, width=2, dash="dot"),
            name="LightGBM Forecast",
        ))
        fig_unc.update_layout(
            **PLOTLY_DARK, height=380,
            title="Demand Forecast + 95% CI (Top 12 States)",
            yaxis_title="Orders (3 months)",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_unc, use_container_width=True)

    # Scenario details table
    st.markdown("### Scenario Details")
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TORNADO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Tornado Diagram — Parameter Sensitivity (±20%)")
    st.caption("How much does total cost change when each parameter shifts by ±20% from baseline?")

    BASE_COST = 1_011_446

    try:
        t_df = tornado_df.sort_values("Range", ascending=True).copy()

        fig_t = go.Figure()

        fig_t.add_trace(go.Bar(
            y=t_df["Parameter"],
            x=(t_df["Low (−20%)"] - BASE_COST) / 1000,
            orientation="h",
            name="−20% change",
            marker_color=BLUE,
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Δ Cost: R$ %{x:.0f}k<extra></extra>",
        ))
        fig_t.add_trace(go.Bar(
            y=t_df["Parameter"],
            x=(t_df["High (+20%)"] - BASE_COST) / 1000,
            orientation="h",
            name="+20% change",
            marker_color=ACCENT,
            opacity=0.85,
            hovertemplate="<b>%{y}</b><br>Δ Cost: R$ %{x:.0f}k<extra></extra>",
        ))

        fig_t.add_vline(x=0, line_color="white", line_width=1.5, opacity=0.4)

        fig_t.update_layout(
            **PLOTLY_DARK,
            barmode="overlay",
            height=380,
            xaxis_title="Change in Total Cost vs. Baseline (R$ thousands)",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, b=50, l=150),
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # Interpretation
        top_param = t_df.iloc[-1]["Parameter"]
        top_range = t_df.iloc[-1]["Range"]
        st.info(
            f"**Key insight**: **{top_param}** is the most influential cost driver "
            f"(R$ {top_range:,.0f} range for ±20% change). "
            f"A 20% increase in demand raises total cost by "
            f"R$ {(t_df.iloc[-1]['High (+20%)'] - BASE_COST):,.0f}."
        )
    except Exception as e:
        st.warning(f"Could not render tornado: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEMAND FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Monthly Demand Trends")

    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        # National trend
        national = (monthly.groupby("order_month")[["orders","revenue"]]
                    .sum().reset_index())

        fig_nat = go.Figure()
        fig_nat.add_trace(go.Scatter(
            x=national["order_month"], y=national["orders"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(255,140,50,0.08)",
            line=dict(color=ACCENT, width=2.5),
            name="Orders",
            hovertemplate="%{x|%b %Y}: <b>%{y:,}</b> orders<extra></extra>",
        ))
        fig_nat.update_layout(
            **PLOTLY_DARK, height=320,
            title="National Monthly Order Volume (2016–2018)",
            yaxis_title="Orders",
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_nat, use_container_width=True)

    with col_f2:
        # Demand share donut
        top_states = demand_df.nlargest(8, "total_forecast")
        other_val  = demand_df["total_forecast"].sum() - top_states["total_forecast"].sum()
        donut_labels = list(top_states["demand_state"]) + ["Others"]
        donut_vals   = list(top_states["total_forecast"]) + [other_val]

        fig_d = go.Figure(go.Pie(
            labels=donut_labels,
            values=donut_vals,
            hole=0.55,
            marker_colors=DC_PALETTE + [MUTED],
            textinfo="label+percent",
            textfont=dict(size=11, family="DM Mono"),
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} orders<br>%{percent}<extra></extra>",
        ))
        fig_d.update_layout(
            **PLOTLY_DARK, height=320,
            title="Demand Share by State",
            showlegend=False,
            margin=dict(t=50, b=10, l=10, r=10),
            annotations=[dict(text="Demand<br>Share",
                              font=dict(size=12, color="#7a8099",
                                        family="DM Mono"),
                              showarrow=False)],
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # Per-state time series selector
    st.markdown("### State-Level Demand")
    selected_states = st.multiselect(
        "Select states to compare",
        options=sorted(monthly["demand_state"].unique()),
        default=["SP", "RJ", "MG", "RS", "PR"],
    )

    if selected_states:
        fig_states = go.Figure()
        colors = DC_PALETTE + [MUTED] * 10
        for k, state in enumerate(selected_states):
            s = monthly[monthly["demand_state"] == state]
            fig_states.add_trace(go.Scatter(
                x=s["order_month"], y=s["orders"],
                mode="lines+markers",
                name=state,
                line=dict(color=colors[k % len(colors)], width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{state}</b> %{{x|%b %Y}}: %{{y:,}} orders<extra></extra>",
            ))
        fig_states.update_layout(
            **PLOTLY_DARK, height=350,
            title=f"Monthly Orders — {', '.join(selected_states)}",
            yaxis_title="Orders",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig_states, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:10px;color:#3a4055;
            text-align:center;padding:8px">
  Supply Chain Network Design · Olist Brazilian E-Commerce Dataset ·
  MILP via PuLP/CBC · LightGBM Demand Forecasting · M2 Applied Statistics
</div>
""", unsafe_allow_html=True)
