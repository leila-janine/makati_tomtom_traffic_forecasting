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
import torch

from prototype_core import (
    HORIZONS,
    MODELS,
    SEEDS,
    build_api_assisted_window,
    build_edge_frame,
    build_segment_frame,
    denormalize_history_speed,
    display_label,
    fetch_live_snapshot,
    load_dataset,
    load_metadata,
    load_summary_metrics,
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


MODEL_LABELS = {
    "persistence": "Persistence baseline",
    "lstm": "LSTM",
    "tgcn": "T GCN",
    "a3tgcn": "A3T GCN",
}
MAKATI_TIMEZONE = ZoneInfo("Asia/Manila")


def secret_or_environment(name: str) -> str:
    try:
        return str(st.secrets[name])
    except (KeyError, FileNotFoundError):
        return os.getenv(name, "")


def metrics_for(model_name: str, horizon: int) -> pd.Series:
    summary = load_summary_metrics()
    selected = summary[
        (summary["model"] == model_name)
        & (summary["horizon_minutes"] == horizon)
    ]
    return selected.iloc[0]


def format_model_table(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame[
        [
            "horizon_minutes",
            "model",
            "runs",
            "parameters",
            "mae_kph_mean",
            "mae_kph_std",
            "rmse_kph_mean",
            "smape_percent_mean",
            "mean_best_epoch",
        ]
    ].copy()
    renamed["model"] = renamed["model"].map(MODEL_LABELS)
    renamed.columns = [
        "Horizon (min)",
        "Model",
        "Runs",
        "Parameters",
        "MAE (km/h)",
        "MAE SD",
        "RMSE (km/h)",
        "sMAPE (%)",
        "Mean best epoch",
    ]
    return renamed.round(3)


with st.sidebar:
    st.header("Forecast controls")
    source_mode = st.radio(
        "Input source",
        ["Observed July test profile", "TomTom API assisted snapshot"],
        help=(
            "The observed profile reproduces the held out test evaluation. "
            "API mode replaces only the latest history step with current flow data."
        ),
    )
    horizon = st.select_slider(
        "Forecast horizon",
        options=list(HORIZONS),
        value=15,
        format_func=lambda value: f"{value} minutes",
    )
    model_name = st.selectbox(
        "Forecast model",
        options=list(MODELS),
        index=3,
        format_func=lambda value: MODEL_LABELS[value],
    )
    ensemble = False
    seed = 42
    if model_name != "persistence":
        ensemble = st.checkbox(
            "Average all five training seeds",
            value=True,
            help="Averages seeds 42, 52, 62, 72, and 82 for a stable prototype output.",
        )
        if not ensemble:
            seed = st.selectbox("Training seed", options=list(SEEDS))
    prefer_cuda = st.checkbox(
        "Use CUDA when available",
        value=True,
        disabled=not torch.cuda.is_available(),
    )


data = load_dataset(horizon)
labels = data["test_labels"].astype(str).tolist()

with st.sidebar:
    sample_label = st.selectbox(
        "Reference test interval",
        labels,
        format_func=display_label,
        help="This chooses the 12 interval history used for observed evaluation and API warm start.",
    )
    sample_index = labels.index(sample_label)
    api_key = ""
    if source_mode == "TomTom API assisted snapshot":
        api_key = st.text_input(
            "TomTom API key",
            value=secret_or_environment("TOMTOM_API_KEY"),
            type="password",
            help="Store this in .streamlit/secrets.toml for deployment.",
        )
        st.caption("One refresh requests a snapshot for each of the 88 segment nodes.")
        fetch_live = st.button("Fetch current traffic", width="stretch")
        if fetch_live:
            with st.spinner("Requesting the 88 TomTom segment points..."):
                try:
                    st.session_state["tomtom_live_snapshot"] = fetch_live_snapshot(api_key)
                    st.session_state["tomtom_live_time"] = datetime.now(MAKATI_TIMEZONE)
                except Exception as exc:
                    st.error(f"Live request failed: {exc}")
    st.divider()
    st.caption("Final experiment configuration")
    st.write("88 nodes · 86 directed edges")
    st.write("12 history intervals · 8 features")
    st.write("May train · June validation · July test")


x = data["x_test"][sample_index].astype(np.float32)
history_speed = denormalize_history_speed(x, data)
current_speed = history_speed[-1].copy()
actual_speed: np.ndarray | None = data["y_test_raw"][sample_index].astype(np.float32)
input_note = f"Held out test interval: {display_label(sample_label)}"
successful_live_nodes = 0

if source_mode == "TomTom API assisted snapshot":
    if "tomtom_live_snapshot" in st.session_state:
        live_time = st.session_state["tomtom_live_time"]
        x, current_speed, successful_live_nodes = build_api_assisted_window(
            x,
            st.session_state["tomtom_live_snapshot"],
            data,
            live_time.hour,
            live_time.minute,
        )
        history_speed[-1] = current_speed
        actual_speed = None
        input_note = (
            f"API snapshot: {live_time:%Y-%m-%d %H:%M} Philippine time · "
            f"{successful_live_nodes}/88 nodes updated"
        )
    else:
        st.info(
            "Enter the TomTom key and select **Fetch current traffic**. Until then, "
            "the selected July reference history is shown."
        )

with st.spinner(f"Running {MODEL_LABELS[model_name]} inference..."):
    forecast_speed, attention, device_name = predict(
        model_name,
        horizon,
        x,
        ensemble=ensemble,
        seed=seed,
        prefer_cuda=prefer_cuda,
    )

segment_frame = build_segment_frame(current_speed, forecast_speed, actual_speed)
edge_frame = build_edge_frame(segment_frame)
model_metrics = metrics_for(model_name, horizon)

st.title("Makati Road Segment Traffic Forecast")
st.caption(
    f"{input_note} · {MODEL_LABELS[model_name]} · {horizon} minute horizon · "
    f"inference device: {device_name.upper()}"
)

mean_current = float(np.mean(current_speed))
mean_forecast = float(np.mean(forecast_speed))
heavy_share = float((segment_frame["traffic_level"] == "Heavy").mean() * 100)
column1, column2, column3, column4 = st.columns(4)
column1.metric("Current mean speed", f"{mean_current:.2f} km/h")
column2.metric(
    f"Forecast at +{horizon} min",
    f"{mean_forecast:.2f} km/h",
    f"{mean_forecast - mean_current:+.2f} km/h",
)
column3.metric("Heavy traffic segments", f"{heavy_share:.1f}%")
column4.metric(
    "Held out test MAE",
    f"{float(model_metrics['mae_kph_mean']):.2f} km/h",
    help="Mean result from the final experiment. Persistence has one deterministic run.",
)

map_tab, analysis_tab, validation_tab, method_tab = st.tabs(
    ["Network map", "Traffic analysis", "Model validation", "Method and limits"]
)

with map_tab:
    st.markdown(
        """
        <div class="legend-row">
          <span class="legend-item"><span class="legend-dot" style="background:#ab1f2e"></span>Heavy, below 25% of limit</span>
          <span class="legend-item"><span class="legend-dot" style="background:#ef602c"></span>Slow, 25% to below 50%</span>
          <span class="legend-item"><span class="legend-dot" style="background:#f5b734"></span>Moderate, 50% to below 75%</span>
          <span class="legend-item"><span class="legend-dot" style="background:#20a368"></span>Free flowing, at least 75%</span>
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
    if actual_speed is not None:
        table_columns.extend(["actual_speed_kph", "absolute_error_kph"])
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

    if actual_speed is not None:
        observed_mae = float(np.mean(np.abs(actual_speed - forecast_speed)))
        observed_rmse = float(np.sqrt(np.mean((actual_speed - forecast_speed) ** 2)))
        st.caption(
            f"Selected interval error: MAE {observed_mae:.3f} km/h and "
            f"RMSE {observed_rmse:.3f} km/h. The formal reported metrics use every July test window."
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

with validation_tab:
    summary = load_summary_metrics().copy()
    selected_horizon = summary[summary["horizon_minutes"] == horizon].copy()
    selected_horizon["Model"] = selected_horizon["model"].map(MODEL_LABELS)
    selected_horizon = selected_horizon.sort_values("mae_kph_mean")
    st.subheader(f"Held out July performance at {horizon} minutes")
    st.bar_chart(
        selected_horizon.set_index("Model"),
        y=["mae_kph_mean", "rmse_kph_mean"],
        height=340,
    )
    st.dataframe(
        format_model_table(selected_horizon),
        hide_index=True,
        width="stretch",
    )
    best = selected_horizon.iloc[0]
    st.info(
        f"The lowest test MAE at this horizon is produced by "
        f"{MODEL_LABELS[str(best['model'])]} at {best['mae_kph_mean']:.3f} km/h. "
        "The prototype reports this comparison directly and does not assume that the graph model must rank first."
    )
    with st.expander("View all final experiment results"):
        st.dataframe(
            format_model_table(summary),
            hide_index=True,
            width="stretch",
        )

with method_tab:
    st.subheader("Prototype configuration")
    st.markdown(
        """
        - The directed graph contains 88 TomTom road segment nodes and 86 ordered road connections.
        - Each input contains 12 consecutive 15 minute history intervals and eight traffic, road, and time features.
        - The available horizons are 15, 30, and 60 minutes.
        - The neural results use five independent training seeds. The prototype can display one seed or their mean prediction.
        - May is the training partition, June is the validation partition, and July is the held out test partition.
        - Normalization parameters were fitted on the May training partition only to prevent information leakage.
        """
    )
    st.subheader("Interpretation limits")
    st.warning(
        "The saved datasets represent aggregate traffic profiles rather than a continuous live archive. "
        "The TomTom API mode is a demonstration that replaces only the newest input interval. "
        "Its preceding eleven intervals come from the selected July test history, so API assisted output "
        "must not be reported as an independently validated live forecast."
    )
    st.caption(
        "Traffic percentages classify forecast speed relative to each segment's speed limit. "
        "They describe the share of graph nodes in each condition, not vehicle volume share."
    )
