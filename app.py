"""
Groundwater Decision Support System (DSS) - Lower Bhavani Basin
================================================================
Integrates 33 years of observed groundwater levels (1993-2025), climate and
LULC drivers, and ML forecasts (2026-2050) under SSP2-4.5 / SSP5-8.5 scenarios
into a decision-support dashboard for researchers and water managers.

Run:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------- paths -----
import sys
if getattr(sys, 'frozen', False):
    BASE = Path(sys._MEIPASS)
    OUT = BASE / "output"
else:
    out_local = Path(__file__).parent / "output"
    if out_local.exists():
        OUT = out_local
    else:
        BASE = Path(__file__).resolve().parent.parent          # folder containing output/
        OUT = BASE / "output"
FUT = OUT / "Future prediction new"

MODELS = ["Random Forest", "Gradient Boosting", "SVR",
          "AdaBoost", "XGBoost", "Extra Trees"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.set_page_config(page_title="HydroInsight · Lower Bhavani Basin",
                   page_icon="💧", layout="wide")


# ---------------------------------------------------------------- data ------
@st.cache_data
def load_data():
    d = {}
    d["annual"] = pd.read_csv(OUT / "GWL_Annual_Summary_1993_2025.csv")
    d["merged"] = pd.read_csv(OUT / "Climate_GWL_Averages_Merged.csv")
    d["monthly"] = pd.read_csv(OUT / "GWL_Monthly_Matrix.csv")
    d["decomp"] = pd.read_csv(OUT / "GWL_Annual_Decomposition.csv")
    d["lulc"] = pd.read_csv(OUT / "LULC_Percentage_Composition_2001_2025.csv")
    d["perf"] = pd.read_csv(FUT / "GWL_Model_Performance_Comparison.csv")
    d["ssp_metrics"] = pd.read_csv(FUT / "GWL_SSP_Scenario_Comparison_Metrics.csv")
    for key, f in [("fc_base", "GWL_Predictions_2026_2050.csv"),
                   ("fc_245", "GWL_Predictions_SSP245_2026_2050.csv"),
                   ("fc_585", "GWL_Predictions_SSP585_2026_2050.csv")]:
        df = pd.read_csv(FUT / f, parse_dates=["Date"])
        df["Year"] = df["Date"].dt.year
        d[key] = df
    return d


@st.cache_data
def load_spatial_data():
    import geopandas as gpd
    meta_path = Path(__file__).parent / "well_metadata.csv"
    obs_path = Path(__file__).parent / "well_observations.csv"
    shp_path_local = Path(__file__).parent / "Lower bhavani_SHP" / "LB_block.shp"
    shp_path_rel = Path(__file__).resolve().parent.parent.parent.parent / "Lower bhavani_SHP" / "LB_block.shp"
    shp_path_abs = Path("D:/Thesis_PhD_06072026/Lower bhavani_SHP/LB_block.shp")
    
    if shp_path_local.exists():
        shp_path = shp_path_local
    elif shp_path_rel.exists():
        shp_path = shp_path_rel
    else:
        shp_path = shp_path_abs
    
    meta = pd.read_csv(meta_path)
    obs = pd.read_csv(obs_path)
    
    # Clip wells to the basin boundary only (remove 'Outside Basin' wells)
    meta = meta[meta['Block_Name'] != 'Outside Basin'].reset_index(drop=True)
    obs = obs[obs['Well No'].isin(meta['Well No'])].reset_index(drop=True)
    
    gdf = None
    if shp_path.exists():
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            gdf['geometry'] = gdf['geometry'].simplify(0.0005, preserve_topology=True)
        except Exception as e:
            st.sidebar.error(f"Error loading shapefile: {e}")
            
    return meta, obs, gdf


def sens_slope(years, values):
    """Median of pairwise slopes (Sen's estimator)."""
    slopes = [(values[j] - values[i]) / (years[j] - years[i])
              for i in range(len(years)) for j in range(i + 1, len(years))]
    return float(np.median(slopes))


def classify_status(depth, hist):
    """Classify current depth against the historical distribution (m bgl)."""
    p25, p50, p75, p90 = np.percentile(hist, [25, 50, 75, 90])
    if depth >= p90:
        return "Critical", "🔴", "Deeper than 90% of historical years"
    if depth >= p75:
        return "Stressed", "🟠", "Deeper than 75% of historical years"
    if depth >= p50:
        return "Watch", "🟡", "Deeper than the historical median"
    return "Safe", "🟢", "Shallower than the historical median"


D = load_data()
annual = D["annual"]
merged = D["merged"]

# Core indicators used across pages
hist_depths = annual["Mean GWL (m)"].values
latest_year = int(annual["Year"].max())
latest_depth = float(annual.loc[annual["Year"] == latest_year, "Mean GWL (m)"].iloc[0])
slope = sens_slope(annual["Year"].values, hist_depths)
status, icon, status_note = classify_status(latest_depth, hist_depths)
best_model = D["perf"].sort_values("Performance Rank").iloc[0]["Model"]

SCEN_MAP = {"Baseline (historical climate)": "fc_base",
            "SSP2-4.5 (moderate emissions)": "fc_245",
            "SSP5-8.5 (high emissions)": "fc_585"}

# ---------------------------------------------------------------- sidebar ---
logo_path = Path(__file__).parent / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=120)
st.sidebar.title("HydroInsight")
st.sidebar.caption("Lower Bhavani Basin · 1993–2050")
page = st.sidebar.radio("Navigate", [
    "1 · Status Overview",
    "2 · Historical Explorer",
    "3 · Forecast Explorer",
    "4 · Driver Analysis",
    "5 · Management Advisor",
    "6 · Spatial Explorer",
])
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Current status ({latest_year}):** {icon} {status}\n\n"
    f"Mean depth: **{latest_depth:.2f} m bgl**\n\n"
    f"Long-term trend: **{slope * 1000:+.1f} mm/yr** "
    f"({'deepening' if slope > 0 else 'recovering'})")
st.sidebar.caption("Depths are metres below ground level (m bgl). "
                   "Larger = deeper = worse.")

# ================================================================ PAGE 1 ====
if page.startswith("1"):
    st.title("Groundwater Status Overview")
    st.caption("Plain-language summary first; technical details in the expanders.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Mean depth {latest_year}", f"{latest_depth:.2f} m",
              f"{latest_depth - float(np.mean(hist_depths)):+.2f} m vs 1993–2025 mean",
              delta_color="inverse")
    c2.metric("Status", f"{icon} {status}", status_note, delta_color="off")
    c3.metric("Long-term trend", f"{slope * 1000:+.1f} mm/yr",
              "Deepening (declining)" if slope > 0 else "Recovering",
              delta_color="inverse" if slope > 0 else "normal")
    fc = D["fc_base"].groupby("Year")[best_model].mean()
    c4.metric("Forecast 2050 (best model)", f"{fc.iloc[-1]:.2f} m",
              f"{fc.iloc[-1] - latest_depth:+.2f} m vs {latest_year}",
              delta_color="inverse")

    # annual series with percentile bands
    p25, p75, p90 = np.percentile(hist_depths, [25, 75, 90])
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=p25, fillcolor="green", opacity=0.07, line_width=0)
    fig.add_hrect(y0=p25, y1=p75, fillcolor="yellow", opacity=0.07, line_width=0)
    fig.add_hrect(y0=p75, y1=p90, fillcolor="orange", opacity=0.10, line_width=0)
    fig.add_hrect(y0=p90, y1=max(hist_depths) * 1.1, fillcolor="red",
                  opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=annual["Year"], y=annual["Mean GWL (m)"],
                             mode="lines+markers", name="Annual mean depth",
                             line=dict(color="#1f77b4")))
    z = np.polyfit(annual["Year"], hist_depths, 1)
    fig.add_trace(go.Scatter(x=annual["Year"], y=np.polyval(z, annual["Year"]),
                             mode="lines", name="Linear trend",
                             line=dict(dash="dash", color="black")))
    fig.update_layout(title="Annual mean groundwater depth with stress bands",
                      yaxis_title="Depth (m bgl)", yaxis_autorange="reversed",
                      height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Y-axis reversed: up = shallower = better. Bands: green Safe, "
               "yellow Watch, orange Stressed, red Critical (historical percentiles).")

    with st.expander("Technical detail — trend statistics"):
        st.markdown(
            f"- **Sen's slope:** {slope:.4f} m/yr (deepening at ~{slope*1000:.1f} mm/yr)\n"
            f"- **Mann-Kendall (1993–2025):** Z = +0.294, p = 0.769 → trend **not "
            f"statistically significant** at α = 0.05 (also non-significant under "
            f"Hamed–Rao modified MK)\n"
            f"- Interpretation: the basin shows a weak deepening tendency dominated "
            f"by inter-annual (monsoon-driven) variability rather than a monotonic "
            f"decline. Full tests: `output/GWL_Trend_Analysis_Summary.xlsx`.")

# ================================================================ PAGE 2 ====
elif page.startswith("2"):
    st.title("Historical Explorer (1993–2025)")

    yr = st.slider("Year range", 1993, 2025, (1993, 2025))
    a = annual[annual["Year"].between(*yr)]

    fig = go.Figure([
        go.Scatter(x=a["Year"], y=a["Min GWL (m)"], line=dict(width=0),
                   showlegend=False),
        go.Scatter(x=a["Year"], y=a["Max GWL (m)"], fill="tonexty",
                   fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                   name="Min–Max range"),
        go.Scatter(x=a["Year"], y=a["Mean GWL (m)"], name="Mean",
                   line=dict(color="#1f77b4", width=2)),
        go.Scatter(x=a["Year"], y=a["Median GWL (m)"], name="Median",
                   line=dict(dash="dot", color="#ff7f0e")),
    ])
    fig.update_layout(title="Annual statistics", yaxis_title="Depth (m bgl)",
                      yaxis_autorange="reversed", height=420)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        m = D["monthly"][D["monthly"]["Year"].between(*yr)]
        hm = px.imshow(m[MONTHS].values, x=MONTHS, y=m["Year"],
                       color_continuous_scale="RdYlBu_r", aspect="auto",
                       labels=dict(color="Depth (m)"),
                       title="Monthly mean depth heatmap")
        st.plotly_chart(hm, use_container_width=True)
        st.caption("Seasonal signal: shallowest after NE monsoon (Nov–Jan), "
                   "deepest pre-monsoon (May–Sep).")
    with col2:
        dc = D["decomp"][D["decomp"]["Year"].between(*yr)]
        fig2 = go.Figure([
            go.Scatter(x=dc["Year"], y=dc["Observed"], name="Observed"),
            go.Scatter(x=dc["Year"], y=dc["Trend"], name="Trend",
                       line=dict(dash="dash")),
        ])
        fig2.update_layout(title="Decomposition: observed vs trend",
                           yaxis_title="Depth (m bgl)",
                           yaxis_autorange="reversed", height=450)
        st.plotly_chart(fig2, use_container_width=True)

    # seasonal recharge indicator
    mm = D["monthly"][D["monthly"]["Year"].between(*yr)]
    recharge = (mm[["May", "Jun", "Jul"]].mean(axis=1)
                - mm[["Nov", "Dec", "Jan"]].mean(axis=1))
    fig3 = px.bar(x=mm["Year"], y=recharge,
                  labels={"x": "Year", "y": "Seasonal recovery (m)"},
                  title="Monsoon recharge effectiveness "
                        "(pre-monsoon depth − post-monsoon depth)")
    fig3.update_traces(marker_color=np.where(recharge > 0, "#2ca02c", "#d62728"))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("Positive bars = water table recovered after the monsoon that year.")

# ================================================================ PAGE 3 ====
elif page.startswith("3"):
    st.title("Forecast Explorer (2026–2050)")

    c1, c2 = st.columns([1, 2])
    with c1:
        scen = st.selectbox("Climate scenario", list(SCEN_MAP))
        models_sel = st.multiselect("Models", MODELS,
                                    default=[best_model, "Gradient Boosting"])
        show_hist = st.checkbox("Show 1993–2025 observed", True)
        agg = st.radio("Resolution", ["Annual", "Monthly"], horizontal=True)
    fcd = D[SCEN_MAP[scen]]

    fig = go.Figure()
    if show_hist:
        fig.add_trace(go.Scatter(x=annual["Year"], y=annual["Mean GWL (m)"],
                                 name="Observed (annual mean)",
                                 line=dict(color="black")))
    for mdl in models_sel:
        if agg == "Annual":
            s = fcd.groupby("Year")[mdl].mean()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=mdl))
        else:
            fig.add_trace(go.Scatter(x=fcd["Date"], y=fcd[mdl], name=mdl,
                                     line=dict(width=1)))
    if models_sel:
        ens = fcd.groupby("Year")[models_sel].mean().mean(axis=1)
        fig.add_trace(go.Scatter(x=ens.index, y=ens.values,
                                 name="Ensemble mean (selected)",
                                 line=dict(color="crimson", dash="dash", width=3)))
    fig.update_layout(title=f"Groundwater depth forecast — {scen}",
                      yaxis_title="Depth (m bgl)", yaxis_autorange="reversed",
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

    # scenario comparison for the best model
    st.subheader("Scenario comparison (annual ensemble of selected models)")
    comp = pd.DataFrame({
        s: D[k].groupby("Year")[models_sel or [best_model]].mean().mean(axis=1)
        for s, k in SCEN_MAP.items()})
    figc = px.line(comp, labels={"value": "Depth (m bgl)", "Year": "Year",
                                 "variable": "Scenario"})
    figc.update_layout(yaxis_autorange="reversed", height=400)
    st.plotly_chart(figc, use_container_width=True)

    with st.expander("Technical detail — model skill and caveats"):
        st.dataframe(D["perf"].round(3), use_container_width=True)
        st.markdown(
            "⚠️ **Caveat:** hold-out R² is negative for all models — the models "
            "do not outperform the historical mean on the test window. Forecasts "
            "should be read as *scenario envelopes*, not precise predictions. "
            "The scenario spread (SSP5-8.5 vs SSP2-4.5 mean drawdown difference "
            "of ~0.01–0.35 m depending on model) is more informative than any "
            "single trajectory.")
        st.dataframe(D["ssp_metrics"].round(3), use_container_width=True)

# ================================================================ PAGE 4 ====
elif page.startswith("4"):
    st.title("Driver Analysis — what controls groundwater levels?")

    corr_vars = st.multiselect(
        "Variables", ["GWL_depth", "Rainfall", "Tmean", "Tmax", "Tmin"],
        default=["GWL_depth", "Rainfall", "Tmean", "Tmax"])
    if len(corr_vars) >= 2:
        cm = merged[corr_vars].corr().round(2)
        st.plotly_chart(px.imshow(cm, text_auto=True, zmin=-1, zmax=1,
                                  color_continuous_scale="RdBu_r",
                                  title="Pearson correlations (annual, 1993–2025)"),
                        use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        xvar = st.selectbox("Driver", ["Rainfall", "Tmean", "Tmax", "Tmin"])
        figs = px.scatter(merged, x=xvar, y="GWL_depth",
                          hover_data=["Year"],
                          labels={"GWL_depth": "Depth (m bgl)"},
                          title=f"GWL depth vs {xvar}")
        b, a = np.polyfit(merged[xvar], merged["GWL_depth"], 1)
        xs = np.linspace(merged[xvar].min(), merged[xvar].max(), 50)
        r = float(np.corrcoef(merged[xvar], merged["GWL_depth"])[0, 1])
        figs.add_trace(go.Scatter(x=xs, y=b * xs + a, mode="lines",
                                  name=f"OLS fit (r = {r:+.2f})",
                                  line=dict(dash="dash", color="crimson")))
        figs.update_layout(yaxis_autorange="reversed")
        st.plotly_chart(figs, use_container_width=True)
    with c2:
        lulc = D["lulc"]
        classes = ["Agriculture", "Forest", "Grassland", "Built_up",
                   "Barren", "Water"]
        figl = px.area(lulc, x="Year", y=classes,
                       title="Land-use composition (%), 2001–2025")
        st.plotly_chart(figl, use_container_width=True)

    st.subheader("LULC ↔ GWL relationships")
    lc = D["lulc"][["GWL_depth", "Rainfall", "Tmean"] + classes].corr().round(2)
    st.plotly_chart(px.imshow(lc, text_auto=True, zmin=-1, zmax=1,
                              color_continuous_scale="RdBu_r",
                              title="LULC × climate × GWL correlation matrix"),
                    use_container_width=True)

    with st.expander("Interpretation notes"):
        bu = D["lulc"]["Built_up"]
        st.markdown(
            f"- **Tmax** correlates positively with depth (r ≈ +0.38*): hotter "
            f"years → higher evapotranspiration and pumping → deeper water table.\n"
            f"- **Rainfall** shows a weak same-year link; recharge response is "
            f"seasonal and mediated by aquifer memory (see lag analysis in "
            f"`output/Climate_GWL_Correlations.xlsx`).\n"
            f"- **Built-up area** grew from {bu.iloc[0]:.1f}% to {bu.iloc[-1]:.1f}% "
            f"while agriculture contracted — urbanisation reduces recharge area.\n"
            f"- Correlation ≠ causation; n is small for LULC epochs (n = 9).")

# ================================================================ PAGE 5 ====
elif page.startswith("5"):
    st.title("Management Advisor")
    st.caption("Rule-based recommendations from status, trend, forecasts and drivers. "
               "Decision-support only — validate with field data before acting.")

    scen = st.selectbox("Planning scenario", list(SCEN_MAP), index=1)
    horizon = st.slider("Planning horizon (year)", 2030, 2050, 2040, step=5)

    fcd = D[SCEN_MAP[scen]]
    ens = fcd.groupby("Year")[MODELS].mean().mean(axis=1)
    fut_depth = float(ens[ens.index <= horizon].iloc[-1])
    fut_status, fut_icon, _ = classify_status(fut_depth, hist_depths)
    ssp_gap = float(D["ssp_metrics"]["Mean Drawdown Increase (m)"].mean())
    bu_growth = float(D["lulc"]["Built_up"].iloc[-1] - D["lulc"]["Built_up"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Now ({latest_year})", f"{icon} {status}", f"{latest_depth:.2f} m bgl",
              delta_color="off")
    c2.metric(f"Projected {horizon}", f"{fut_icon} {fut_status}",
              f"{fut_depth:.2f} m bgl ({fut_depth - latest_depth:+.2f} m)",
              delta_color="off")
    c3.metric("SSP5-8.5 penalty", f"+{ssp_gap:.2f} m",
              "extra mean drawdown vs SSP2-4.5", delta_color="off")

    st.markdown("---")
    recs = []
    if status in ("Critical", "Stressed") or fut_status in ("Critical", "Stressed"):
        recs.append(("HIGH", "Managed aquifer recharge (MAR)",
                     "Prioritise check dams, percolation ponds and recharge shafts in "
                     "over-exploited blocks; target NE-monsoon runoff capture (Oct–Dec)."))
        recs.append(("HIGH", "Regulate abstraction",
                     "Enforce well-spacing/permit norms and seasonal pumping limits in "
                     "blocks where depth exceeds the 75th historical percentile."))
    if slope > 0:
        recs.append(("MEDIUM", "Demand-side management",
                     "Promote micro-irrigation and shift water-intensive crops in the "
                     "command area; the basin shows a deepening tendency of "
                     f"~{slope*1000:.0f} mm/yr."))
    if ssp_gap > 0.05:
        recs.append(("MEDIUM", "Climate-proof planning",
                     f"Plan infrastructure against the SSP5-8.5 envelope (+{ssp_gap:.2f} m "
                     "mean extra drawdown) rather than the moderate scenario."))
    if bu_growth > 0.5:
        recs.append(("MEDIUM", "Protect recharge zones",
                     f"Built-up area rose {bu_growth:+.1f} pp since 2001. Zone and protect "
                     "high-infiltration areas; mandate rainwater harvesting in new builds."))
    recs.append(("ONGOING", "Strengthen monitoring",
                 "Maintain the observation-well network, add telemetry in data-sparse "
                 "blocks, and re-train forecast models annually (current model skill is "
                 "low — negative hold-out R²)."))

    colors = {"HIGH": "🔴", "MEDIUM": "🟠", "ONGOING": "🔵"}
    for pr, title, body in recs:
        st.markdown(f"**{colors[pr]} [{pr}] {title}** — {body}")

    st.markdown("---")
    st.subheader("What-if: rainfall sensitivity")
    dr = st.slider("Assumed change in annual rainfall (%)", -30, 30, 0, step=5)
    X = np.column_stack([merged["Rainfall"], merged["Tmean"],
                         np.ones(len(merged))])
    coef, *_ = np.linalg.lstsq(X, merged["GWL_depth"], rcond=None)
    d_depth = coef[0] * merged["Rainfall"].mean() * dr / 100
    st.metric("Estimated change in mean depth",
              f"{d_depth:+.2f} m",
              "shallower (better)" if d_depth < 0 else
              ("deeper (worse)" if d_depth > 0 else "no change"),
              delta_color="off")
    st.caption("Linear regression of annual GWL depth on rainfall and mean "
               "temperature (1993–2025). Indicative only — the same-year "
               "rainfall–GWL correlation is weak (r ≈ 0.17, n.s.).")

# ================================================================ PAGE 6 ====
else:
    st.title("Spatial Explorer & Well-Level Forecasts")
    st.caption("Interactive map of observation wells. Click a well on the map or select from the dropdown to view historical trends and projections.")
    
    # Load spatial data
    meta_df, obs_df, blocks_gdf = load_spatial_data()
    
    # Sidebar filters for well selection
    st.sidebar.subheader("Well Filters")
    
    # 1. Block filter (only show blocks inside the basin boundary)
    blocks_list = sorted([b for b in meta_df["Block_Name"].unique() if b != "Outside Basin"])
    block_filter = st.sidebar.selectbox("Filter by Block", ["All Blocks"] + blocks_list)
    
    # 2. Filter metadata
    filtered_meta = meta_df.copy()
    if block_filter != "All Blocks":
        filtered_meta = filtered_meta[filtered_meta["Block_Name"] == block_filter]
        
    well_types = sorted(filtered_meta["Well Type"].unique())
    well_type_filter = st.sidebar.selectbox("Filter by Well Type", ["All Types"] + well_types)
    if well_type_filter != "All Types":
        filtered_meta = filtered_meta[filtered_meta["Well Type"] == well_type_filter]
        
    if filtered_meta.empty:
        st.warning("No wells match the selected filters.")
    else:
        # 3. Synchronized Well Selection
        if "selected_well" not in st.session_state or st.session_state["selected_well"] not in filtered_meta["Well No"].values:
            st.session_state["selected_well"] = filtered_meta["Well No"].iloc[0]
            
        well_options = list(filtered_meta["Well No"].values)
        try:
            default_idx = well_options.index(st.session_state["selected_well"])
        except ValueError:
            default_idx = 0
            st.session_state["selected_well"] = well_options[0]
            
        selected_well_id = st.sidebar.selectbox(
            "Select Well",
            well_options,
            index=default_idx,
            key="well_select_dropdown"
        )
        
        if selected_well_id != st.session_state["selected_well"]:
            st.session_state["selected_well"] = selected_well_id
            
        # Layout columns
        col_map, col_details = st.columns([1, 1])
        
        with col_map:
            st.write("### Observation Wells Map")
            
            map_style = st.selectbox(
                "Map Style",
                ["open-street-map", "carto-positron", "white-bg"],
                index=1
            )
            
            # Map wells to their status for color-coding
            latest_obs = obs_df.loc[obs_df.groupby('Well No')['Year'].idxmax()]
            
            status_color_map = {
                "Safe": "green",
                "Watch": "gold",
                "Stressed": "darkorange",
                "Critical": "red",
                "Unknown": "gray"
            }
            
            well_status_dict = {}
            for _, row in latest_obs.iterrows():
                wn = row['Well No']
                wl = row['Water Level']
                well_all_vals = obs_df[obs_df['Well No'] == wn]['Water Level'].values
                
                # Inline classification logic
                if len(well_all_vals) < 5:
                    w_status = "Unknown"
                else:
                    p25, p50, p75, p90 = np.percentile(well_all_vals, [25, 50, 75, 90])
                    if wl >= p90:
                        w_status = "Critical"
                    elif wl >= p75:
                        w_status = "Stressed"
                    elif wl >= p50:
                        w_status = "Watch"
                    else:
                        w_status = "Safe"
                well_status_dict[wn] = w_status
                
            filtered_meta['Status'] = filtered_meta['Well No'].map(well_status_dict).fillna('Unknown')
            filtered_meta['Color'] = filtered_meta['Status'].map(status_color_map).fillna('gray')
            
            fig_map = go.Figure()
            
            # 1. Add blocks boundaries if shapefile loaded
            if blocks_gdf is not None:
                geojson = blocks_gdf.__geo_interface__
                fig_map.add_trace(go.Choroplethmapbox(
                    geojson=geojson,
                    locations=blocks_gdf.index,
                    z=[1] * len(blocks_gdf),
                    colorscale=[[0, "rgba(120,120,120,0.12)"], [1, "rgba(120,120,120,0.12)"]],
                    showscale=False,
                    hoverinfo="text",
                    text=blocks_gdf["Block_Name"],
                    name="Basin Blocks"
                ))
            
            # 2. Add well points grouped by status
            for status_name, color in status_color_map.items():
                status_meta = filtered_meta[filtered_meta['Status'] == status_name]
                if status_meta.empty:
                    continue
                    
                fig_map.add_trace(go.Scattermapbox(
                    lat=status_meta['Latitude'],
                    lon=status_meta['Longitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=9,
                        color=color,
                        opacity=0.85
                    ),
                    text=status_meta['Well No'],
                    customdata=status_meta['Well No'],
                    name=f"Status: {status_name}",
                    hovertemplate="<b>Well: %{text}</b><br>Block: " + status_meta['Block_Name'] + "<br>Status: " + status_name + "<extra></extra>"
                ))
                
            # 3. Highlight selected well
            sel_meta = filtered_meta[filtered_meta['Well No'] == st.session_state["selected_well"]]
            if not sel_meta.empty:
                # Black outer circle for outline effect
                fig_map.add_trace(go.Scattermapbox(
                    lat=sel_meta['Latitude'],
                    lon=sel_meta['Longitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=15,
                        color='black',
                        opacity=0.9
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                # Cyan inner circle
                fig_map.add_trace(go.Scattermapbox(
                    lat=sel_meta['Latitude'],
                    lon=sel_meta['Longitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=9,
                        color='cyan',
                        opacity=1.0
                    ),
                    text=[f"📍 {st.session_state['selected_well']}"],
                    customdata=sel_meta['Well No'],
                    name="Selected Well",
                    hovertemplate="<b>Selected Well: %{customdata}</b><br>Block: " + sel_meta['Block_Name'].iloc[0] + "<extra></extra>"
                ))
                
            if not filtered_meta.empty:
                center_lat = filtered_meta['Latitude'].mean()
                center_lon = filtered_meta['Longitude'].mean()
            else:
                center_lat = 11.4
                center_lon = 77.3
                
            fig_map.update_layout(
                mapbox=dict(
                    style=map_style,
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=9.0 if block_filter == "All Blocks" else 10.5
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=550,
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=0.02,
                    bgcolor="rgba(255, 255, 255, 0.7)"
                )
            )
            
            map_event = st.plotly_chart(
                fig_map, 
                use_container_width=True, 
                on_select="rerun", 
                key="well_map"
            )
            
            if map_event and "selection" in map_event and "points" in map_event["selection"] and len(map_event["selection"]["points"]) > 0:
                clicked_pt = map_event["selection"]["points"][0]
                clicked_well = clicked_pt.get("customdata")
                if clicked_well and clicked_well in filtered_meta["Well No"].values:
                    if clicked_well != st.session_state["selected_well"]:
                        st.session_state["selected_well"] = clicked_well
                        st.rerun()
                        
        with col_details:
            curr_well = st.session_state["selected_well"]
            well_info = meta_df[meta_df["Well No"] == curr_well].iloc[0]
            
            st.write(f"### Well Details: {curr_well}")
            
            well_obs = obs_df[obs_df["Well No"] == curr_well].copy()
            well_obs_sorted = well_obs.sort_values(["Year", "Month"]).reset_index(drop=True)
            
            latest_row = well_obs_sorted.iloc[-1] if not well_obs_sorted.empty else None
            if latest_row is not None:
                latest_val = latest_row["Water Level"]
                latest_yr = int(latest_row["Year"])
                well_all_vals = well_obs["Water Level"].values
                well_status, well_icon, well_desc = classify_status(latest_val, well_all_vals)
                
                well_annual = well_obs.groupby("Year")["Water Level"].mean()
                if len(well_annual) >= 2:
                    well_slope_val = sens_slope(well_annual.index.values, well_annual.values)
                else:
                    well_slope_val = 0.0
                    
                c1, c2, c3 = st.columns(3)
                c1.metric("Block / District", f"{well_info['Block_Name']}", f"{well_info['District']}")
                c2.metric(f"Latest Depth ({latest_yr})", f"{latest_val:.2f} m", well_icon + " " + well_status, delta_color="off")
                c3.metric("Local Trend", f"{well_slope_val * 1000:+.1f} mm/yr", "Deepening" if well_slope_val > 0 else "Recovering", delta_color="inverse" if well_slope_val > 0 else "normal")
                
                with st.expander("Show Well Metadata"):
                    st.markdown(
                        f"- **Well ID:** `{well_info['Well No']}`\n"
                        f"- **District:** {well_info['District']}\n"
                        f"- **Block:** {well_info['Block_Name']}\n"
                        f"- **Coordinates:** {well_info['Latitude']:.6f}° N, {well_info['Longitude']:.6f}° E\n"
                        f"- **Well Type:** {well_info['Well Type']}\n"
                        f"- **Data Span:** {well_annual.index.min()} to {well_annual.index.max()} ({len(well_obs)} monthly records)"
                    )
                    
                st.write("#### Historical observations & climate-scenario forecasts (to 2050)")
                
                well_hist_mean = well_annual.mean()
                basin_hist_mean = annual["Mean GWL (m)"].mean()
                offset = well_hist_mean - basin_hist_mean
                
                fc_years = D["fc_base"].groupby("Year")["Date"].first().index.values
                
                fc_base_vals = D["fc_base"].groupby("Year")[MODELS].mean().mean(axis=1).values + offset
                fc_245_vals = D["fc_245"].groupby("Year")[MODELS].mean().mean(axis=1).values + offset
                fc_585_vals = D["fc_585"].groupby("Year")[MODELS].mean().mean(axis=1).values + offset
                
                trend_years = np.arange(well_annual.index.min(), 2051)
                if len(well_annual) >= 2:
                    b_fit, a_fit = np.polyfit(well_annual.index.values, well_annual.values, 1)
                    trend_vals = b_fit * trend_years + a_fit
                else:
                    trend_vals = np.array([well_hist_mean] * len(trend_years))
                    
                fig_chart = go.Figure()
                
                obs_dates = pd.to_datetime(well_obs_sorted['Year'].astype(str) + '-' + well_obs_sorted['Month'].astype(str) + '-01')
                fig_chart.add_trace(go.Scatter(
                    x=obs_dates,
                    y=well_obs_sorted['Water Level'],
                    mode='lines',
                    line=dict(color='rgba(31,119,180,0.25)', width=1),
                    name='Monthly observations',
                    showlegend=True
                ))
                
                fig_chart.add_trace(go.Scatter(
                    x=pd.to_datetime(well_annual.index.astype(str) + '-07-01'),
                    y=well_annual.values,
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=2),
                    name='Annual mean depth'
                ))
                
                fig_chart.add_trace(go.Scatter(
                    x=pd.to_datetime(trend_years.astype(str) + '-07-01'),
                    y=trend_vals,
                    mode='lines',
                    line=dict(color='black', dash='dash', width=1.5),
                    name='Historical linear trend'
                ))
                
                fig_chart.add_trace(go.Scatter(
                    x=pd.to_datetime(fc_years.astype(str) + '-07-01'),
                    y=fc_base_vals,
                    mode='lines',
                    line=dict(color='green', dash='dot', width=2),
                    name='Future: Baseline (Hist Climate)'
                ))
                
                fig_chart.add_trace(go.Scatter(
                    x=pd.to_datetime(fc_years.astype(str) + '-07-01'),
                    y=fc_245_vals,
                    mode='lines',
                    line=dict(color='orange', width=2.5),
                    name='Future: SSP2-4.5 (Moderate)'
                ))
                
                fig_chart.add_trace(go.Scatter(
                    x=pd.to_datetime(fc_years.astype(str) + '-07-01'),
                    y=fc_585_vals,
                    mode='lines',
                    line=dict(color='red', width=2.5),
                    name='Future: SSP5-8.5 (High Emissions)'
                ))
                
                fig_chart.update_layout(
                    title=f"Groundwater Depth Forecast Envelope: Well {curr_well}",
                    xaxis_title="Year",
                    yaxis_title="Depth (m bgl)",
                    yaxis_autorange="reversed",
                    height=400,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.35,
                        xanchor="center",
                        x=0.5
                    )
                )
                
                st.plotly_chart(fig_chart, use_container_width=True)
                st.caption("Y-axis reversed: up = shallower = better. Shaded/light lines show monthly data, solid blue shows annual mean. Projections are downscaled relative drawdown scenarios.")
            else:
                st.info("Select a well with observations to view historical charts.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Author:**  \n"
    "Harish M, PhD Scholar  \n\n"
    "**Acknowledgements:**  \n"
    "* State Ground and Surface Water Resource Data Centre, Chennai  \n"
    "* Tamil Nadu Agricultural University  \n\n"
    "**Data Sources:**  \n"
    "Observation wells, NASA POWER, LULC rasters, CMIP6 SSP projections"
)
