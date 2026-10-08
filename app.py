"""Interactive TomTom traffic forecasting prototype for the Makati study route."""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from prototype_core import (
    HORIZONS,
    build_api_assisted_window,
    build_edge_frame,
    build_segment_frame,
    denormalize_history_speed,
    fetch_live_snapshot,
    load_dataset,
    load_metadata,
    predict,
)


st.set_page_config(
    page_title="Makati Traffic Forecast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 0.8rem;
        padding: 0.75rem 1rem;
        background: rgba(120, 120, 120, 0.04);
    }
    .legend-row {display:flex; gap:1.1rem; flex-wrap:wrap; margin:0.4rem 0 0.9rem;}
    .legend-item {display:flex; align-items:center; gap:0.35rem; font-size:0.9rem;}
    .legend-dot {width:0.8rem; height:0.8rem; border-radius:50%; display:inline-block;}
    </style>
    """,
    unsafe_allow_html=True,
)


MAKATI_TIMEZONE = ZoneInfo("Asia/Manila")
AUTO_REFRESH_MILLISECONDS = 15 * 60 * 1000


def secret_or_environment(name: str) -> str:
    try:
        return str(st.secrets[name])
    except (KeyError, FileNotFoundError):
        return os.getenv(name, "")


@st.cache_data(ttl=15 * 60, show_spinner=False)
def cached_live_snapshot(api_key: str) -> tuple[pd.DataFrame, datetime]:
    """Fetch one TomTom snapshot and reuse it for at most 15 minutes."""
    return fetch_live_snapshot(api_key), datetime.now(MAKATI_TIMEZONE)


with st.sidebar:
    st.header("Forecast controls")
    horizon = st.select_slider(
        "Forecast horizon",
        options=list(HORIZONS),
        value=15,
        format_func=lambda value: f"{value} minutes",
    )
    st.caption("TomTom traffic updates automatically every 15 minutes.")


data = load_dataset(horizon)
sample_index = len(data["x_test"]) - 1
x = data["x_test"][sample_index].astype(np.float32)
history_speed = denormalize_history_speed(x, data)
current_speed = history_speed[-1].copy()
actual_speed: np.ndarray | None = None
api_key = secret_or_environment("TOMTOM_API_KEY").strip()

if not api_key:
    st.error(
        "The TomTom API key is not configured. Add TOMTOM_API_KEY to the "
        "Streamlit application secrets, then restart the application."
    )
    st.stop()

st_autorefresh(
    interval=AUTO_REFRESH_MILLISECONDS,
    limit=None,
    key="tomtom_traffic_auto_refresh",
)

with st.spinner("Loading current TomTom traffic..."):
    try:
        live_snapshot, live_time = cached_live_snapshot(api_key)
    except Exception as exc:
        st.error(f"TomTom traffic update failed: {exc}")
        st.stop()

x, current_speed, successful_live_nodes = build_api_assisted_window(
    x,
    live_snapshot,
    data,
    live_time.hour,
    live_time.minute,
)
history_speed[-1] = current_speed

model_name = "a3tgcn"
with st.spinner("Running A3T GCN inference..."):
    forecast_speed, attention, device_name = predict(
        model_name,
        horizon,
        x,
        ensemble=True,
        seed=42,
        prefer_cuda=True,
    )

segment_frame = build_segment_frame(current_speed, forecast_speed, actual_speed)
edge_frame = build_edge_frame(segment_frame)

st.title("Makati Road Segment Traffic Forecast")
st.caption(
    f"Updated {live_time:%B %d, %Y at %I:%M %p} Philippine time · "
    f"{successful_live_nodes}/88 road segments received · "
    f"A3T GCN {horizon} minute forecast"
)

mean_current = float(np.mean(current_speed))
mean_forecast = float(np.mean(forecast_speed))
column1, column2 = st.columns(2)
column1.metric("Current mean speed", f"{mean_current:.1f} km/h")
column2.metric(
    f"Forecast at +{horizon} min",
    f"{mean_forecast:.1f} km/h",
    f"{mean_forecast - mean_current:+.1f} km/h",
)

map_tab, analysis_tab = st.tabs(["Network map", "Traffic analysis"])

with map_tab:
    st.markdown(
        """
        <div class="legend-row">
          <span class="legend-item"><span class="legend-dot" style="background:#ab1f2e"></span>Heavy, below 10 km/h</span>
          <span class="legend-item"><span class="legend-dot" style="background:#ef602c"></span>Slow, 10 to below 20 km/h</span>
          <span class="legend-item"><span class="legend-dot" style="background:#f5b734"></span>Moderate, 20 to below 30 km/h</span>
          <span class="legend-item"><span class="legend-dot" style="background:#20a368"></span>Free flowing, at least 30 km/h</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nodes, graph_edges = load_metadata()
    map_data = segment_frame.rename(
        columns={
            "street_name": "street",
            "route_name": "route",
            "forecast_speed_kph": "forecast",
            "current_speed_kph": "current",
            "speed_limit_kph": "speed_limit",
            "traffic_level": "traffic",
        }
    )
    view = pdk.ViewState(
        latitude=float(nodes["latitude"].mean()),
        longitude=float(nodes["longitude"].mean()),
        zoom=13.8,
        pitch=35,
    )
    line_layer = pdk.Layer(
        "PathLayer",
        data=edge_frame,
        get_path="path",
        get_color="color",
        get_width=8,
        width_min_pixels=3,
        pickable=False,
    )
    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=24,
        radius_min_pixels=4,
        radius_max_pixels=12,
        stroked=True,
        get_line_color=[245, 245, 245, 220],
        line_width_min_pixels=1,
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_provider="carto",
            map_style=pdk.map_styles.CARTO_DARK,
            initial_view_state=view,
            layers=[line_layer, node_layer],
            tooltip={
                "html": (
                    "<b>{street}</b><br/>Route: {route}<br/>"
                    "Current: {current} km/h<br/>Forecast: {forecast} km/h<br/>"
                    "Limit: {speed_limit} km/h<br/>Class: {traffic}"
                )
            },
        ),
        width="stretch",
        height=610,
    )

    st.subheader("Directed road network graph")
    st.caption(
        "Nodes are ordered from left to right within each travel direction. "
        "Each connecting line represents a directed downstream road edge."
    )
    topology_nodes = segment_frame.copy()
    topology_nodes["graph_x"] = topology_nodes["route_position"].astype(float)
    topology_nodes["graph_y"] = np.where(
        topology_nodes["route_name"].str.startswith("Other direction"), 0.0, 1.0
    )
    topology_nodes["direction"] = np.where(
        topology_nodes["graph_y"] == 1.0,
        "EDSA to Chino Roces",
        "Chino Roces to EDSA",
    )

    source_layout = topology_nodes[
        ["node_index", "graph_x", "graph_y"]
    ].rename(
        columns={
            "node_index": "source_index",
            "graph_x": "source_x",
            "graph_y": "source_y",
        }
    )
    destination_layout = topology_nodes[
        ["node_index", "graph_x", "graph_y"]
    ].rename(
        columns={
            "node_index": "destination_index",
            "graph_x": "destination_x",
            "graph_y": "destination_y",
        }
    )
    topology_edges = graph_edges.merge(source_layout, on="source_index").merge(
        destination_layout, on="destination_index"
    )

    traffic_scale = alt.Scale(
        domain=["Heavy", "Slow", "Moderate", "Free flowing"],
        range=["#ab1f2e", "#ef602c", "#f5b734", "#20a368"],
    )
    edge_chart = (
        alt.Chart(topology_edges)
        .mark_rule(color="#8a94a6", strokeWidth=2, opacity=0.65)
        .encode(
            x=alt.X("source_x:Q", title="Ordered road segment position"),
            x2="destination_x:Q",
            y=alt.Y(
                "source_y:Q",
                title="Travel direction",
                scale=alt.Scale(domain=[-0.35, 1.35]),
                axis=alt.Axis(
                    values=[0, 1],
                    labelExpr="datum.value == 1 ? 'EDSA → Chino' : 'Chino → EDSA'",
                ),
            ),
            y2="destination_y:Q",
        )
    )
    node_chart = (
        alt.Chart(topology_nodes)
        .mark_circle(size=115, stroke="white", strokeWidth=0.7)
        .encode(
            x=alt.X("graph_x:Q", title="Ordered road segment position"),
            y=alt.Y(
                "graph_y:Q",
                title="Travel direction",
                scale=alt.Scale(domain=[-0.35, 1.35]),
                axis=alt.Axis(
                    values=[0, 1],
                    labelExpr="datum.value == 1 ? 'EDSA → Chino' : 'Chino → EDSA'",
                ),
            ),
            color=alt.Color(
                "traffic_level:N",
                title="Forecast condition",
                scale=traffic_scale,
                sort=["Heavy", "Slow", "Moderate", "Free flowing"],
            ),
            tooltip=[
                alt.Tooltip("node_index:Q", title="Node"),
                alt.Tooltip("street_name:N", title="Street"),
                alt.Tooltip("direction:N", title="Direction"),
                alt.Tooltip("forecast_speed_kph:Q", title="Forecast km/h", format=".2f"),
                alt.Tooltip("traffic_level:N", title="Condition"),
            ],
        )
    )
    st.altair_chart(
        (edge_chart + node_chart).properties(height=330),
        width="stretch",
    )

with analysis_tab:
    st.subheader("Network speed trajectory")
    average_history = history_speed.mean(axis=1)
    history_labels = [f"t-{15 * (11 - index)}" for index in range(12)]
    trajectory = pd.DataFrame(
        {"Interval": history_labels, "Mean speed (km/h)": average_history}
    )
    trajectory = pd.concat(
        [
            trajectory,
            pd.DataFrame(
                {
                    "Interval": [f"t+{horizon}"],
                    "Mean speed (km/h)": [mean_forecast],
                }
            ),
        ],
        ignore_index=True,
    )
    st.line_chart(trajectory, x="Interval", y="Mean speed (km/h)", height=300)

    st.subheader("Traffic percentage by forecast condition")
    class_order = ["Heavy", "Slow", "Moderate", "Free flowing"]
    percentages = (
        segment_frame["traffic_level"]
        .value_counts(normalize=True)
        .reindex(class_order, fill_value=0)
        .mul(100)
        .rename_axis("Traffic condition")
        .reset_index(name="Road segments (%)")
    )
    left, right = st.columns([1, 2])
    left.dataframe(percentages.round(2), hide_index=True, width="stretch")
    right.bar_chart(
        percentages.set_index("Traffic condition"),
        y="Road segments (%)",
        height=275,
    )

    st.subheader("Road segment forecast table")
    table_columns = [
        "node_index",
        "street_name",
        "route_name",
        "current_speed_kph",
        "forecast_speed_kph",
        "change_kph",
        "speed_limit_kph",
        "traffic_level",
    ]
    result_table = segment_frame[table_columns].sort_values("forecast_speed_kph")
    st.dataframe(
        result_table.round(2),
        hide_index=True,
        width="stretch",
        height=430,
    )
    st.download_button(
        "Download this forecast as CSV",
        data=result_table.to_csv(index=False).encode("utf-8"),
        file_name=f"makati_{model_name}_{horizon}min_forecast.csv",
        mime="text/csv",
    )

    if attention is not None:
        st.subheader("A3T GCN temporal attention")
        attention_by_step = attention.mean(axis=0)
        attention_frame = pd.DataFrame(
            {
                "History interval": history_labels,
                "Mean attention weight": attention_by_step,
            }
        )
        st.bar_chart(
            attention_frame,
            x="History interval",
            y="Mean attention weight",
            height=275,
        )
