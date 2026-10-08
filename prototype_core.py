"""Data loading, inference, mapping, and live API helpers for the prototype."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch

from tomtom_models import build_model


BASE_DIR = Path(__file__).resolve().parent
PREPARED_DIR = BASE_DIR / "assets" / "prepared"
CHECKPOINT_DIR = BASE_DIR / "assets" / "checkpoints"
RESULTS_DIR = BASE_DIR / "assets" / "results"
HORIZONS = (15, 30, 60)
MODELS = ("persistence", "lstm", "tgcn", "a3tgcn")
SEEDS = (42, 52, 62, 72, 82)


@lru_cache(maxsize=3)
def load_dataset(horizon: int) -> dict[str, Any]:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported forecast horizon: {horizon}")
    path = PREPARED_DIR / f"tomtom_a3tgcn_{horizon}min.npz"
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


@lru_cache(maxsize=1)
def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(PREPARED_DIR / "node_metadata.csv")
    edges = pd.read_csv(PREPARED_DIR / "graph_edges.csv")
    nodes["street_name"] = nodes["street_name"].fillna("Unnamed road segment")
    nodes["segment_id"] = nodes["segment_id"].astype(str)
    return nodes, edges


@lru_cache(maxsize=1)
def load_summary_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "summary_metrics.csv")


def choose_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=60)
def load_trained_model(
    model_name: str, horizon: int, seed: int, device_name: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = CHECKPOINT_DIR / f"{model_name}_{horizon}min_seed{seed}.pt"
    checkpoint = torch.load(
        path, map_location=torch.device(device_name), weights_only=False
    )
    model = build_model(
        checkpoint["model_name"],
        int(checkpoint["input_size"]),
        int(checkpoint["hidden_size"]),
        float(checkpoint["dropout"]),
    ).to(device_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def persistence_prediction(x: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    index = int(data["speed_feature_index"])
    return (
        x[-1, :, index] * float(data["feature_std"][index])
        + float(data["feature_mean"][index])
    ).astype(np.float32)


def predict(
    model_name: str,
    horizon: int,
    x: np.ndarray,
    ensemble: bool = True,
    seed: int = 42,
    prefer_cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    data = load_dataset(horizon)
    if model_name == "persistence":
        return persistence_prediction(x, data), None, "cpu"

    device = choose_device(prefer_cuda)
    seeds = SEEDS if ensemble else (seed,)
    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    adjacency = torch.as_tensor(
        data["adjacency"], dtype=torch.float32, device=device
    )
    predictions: list[np.ndarray] = []
    attentions: list[np.ndarray] = []
    with torch.inference_mode():
        for run_seed in seeds:
            model, checkpoint = load_trained_model(
                model_name, horizon, run_seed, device.type
            )
            normalized, attention = model(x_tensor, adjacency)
            raw = (
                normalized.detach().cpu().numpy()[0]
                * float(checkpoint["target_std"])
                + float(checkpoint["target_mean"])
            )
            predictions.append(raw)
            if attention is not None:
                attentions.append(attention.detach().cpu().numpy()[0])

    prediction = np.mean(predictions, axis=0).astype(np.float32)
    attention_mean = (
        np.mean(attentions, axis=0).astype(np.float32) if attentions else None
    )
    return prediction, attention_mean, device.type


def denormalize_history_speed(x: np.ndarray, data: dict[str, Any]) -> np.ndarray:
    index = int(data["speed_feature_index"])
    return (
        x[:, :, index] * float(data["feature_std"][index])
        + float(data["feature_mean"][index])
    )


def traffic_class(speed: float) -> str:
    """Classify a segment using fixed forecast speed thresholds in km/h."""
    if speed < 10.0:
        return "Heavy"
    if speed < 20.0:
        return "Slow"
    if speed < 30.0:
        return "Moderate"
    return "Free flowing"


def traffic_color(label: str) -> list[int]:
    return {
        "Heavy": [171, 31, 46, 225],
        "Slow": [239, 96, 44, 225],
        "Moderate": [245, 183, 52, 225],
        "Free flowing": [32, 163, 104, 225],
    }[label]


def build_segment_frame(
    current: np.ndarray,
    forecast: np.ndarray,
    actual: np.ndarray | None = None,
) -> pd.DataFrame:
    nodes, _ = load_metadata()
    frame = nodes.copy()
    frame["current_speed_kph"] = current
    frame["forecast_speed_kph"] = forecast
    frame["change_kph"] = frame["forecast_speed_kph"] - frame["current_speed_kph"]
    frame["traffic_level"] = frame["forecast_speed_kph"].map(traffic_class)
    frame["color"] = frame["traffic_level"].map(traffic_color)
    if actual is not None:
        frame["actual_speed_kph"] = actual
        frame["absolute_error_kph"] = np.abs(actual - forecast)
    return frame


def build_edge_frame(segment_frame: pd.DataFrame) -> pd.DataFrame:
    nodes, edges = load_metadata()
    by_index = nodes.set_index("node_index")
    records = []
    for row in edges.itertuples(index=False):
        source = by_index.loc[row.source_index]
        destination = by_index.loc[row.destination_index]
        destination_result = segment_frame.loc[
            segment_frame["node_index"] == row.destination_index
        ].iloc[0]
        records.append(
            {
                "route_name": row.route_name,
                "path": [
                    [float(source.longitude), float(source.latitude)],
                    [float(destination.longitude), float(destination.latitude)],
                ],
                "color": destination_result["color"],
                "forecast_speed_kph": float(
                    destination_result["forecast_speed_kph"]
                ),
            }
        )
    return pd.DataFrame(records)


def _fetch_one_node(row: Any, api_key: str) -> dict[str, Any]:
    endpoint = (
        "https://api.tomtom.com/traffic/services/4/"
        "flowSegmentData/absolute/10/json"
    )
    result: dict[str, Any] = {"node_index": int(row.node_index), "success": False}
    try:
        response = requests.get(
            endpoint,
            params={
                "point": f"{row.latitude},{row.longitude}",
                "unit": "KMPH",
                "openLr": "false",
                "key": api_key,
            },
            timeout=12,
        )
        response.raise_for_status()
        flow = response.json()["flowSegmentData"]
        result.update(
            {
                "success": True,
                "current_speed_kph": float(flow["currentSpeed"]),
                "free_flow_speed_kph": float(flow["freeFlowSpeed"]),
                "current_travel_time_s": float(flow["currentTravelTime"]),
                "free_flow_travel_time_s": float(flow["freeFlowTravelTime"]),
                "road_closure": bool(flow.get("roadClosure", False)),
            }
        )
    except Exception as exc:  # The caller reports the aggregate failure count.
        result["error"] = str(exc)
    return result


def fetch_live_snapshot(
    api_key: str, max_workers: int = 8
) -> pd.DataFrame:
    if not api_key.strip():
        raise ValueError("A TomTom API key is required for live mode.")
    nodes, _ = load_metadata()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_fetch_one_node, row, api_key)
            for row in nodes.itertuples(index=False)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    return pd.DataFrame(records).sort_values("node_index").reset_index(drop=True)


def build_api_assisted_window(
    base_x: np.ndarray,
    live: pd.DataFrame,
    data: dict[str, Any],
    local_hour: int,
    local_minute: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update the latest history step with an API snapshot.

    The remaining eleven steps stay sourced from the selected observed July
    history. This is a warm start and not a replacement for a live data archive.
    """
    x = base_x.copy()
    feature_mean = data["feature_mean"].astype(np.float32)
    feature_std = data["feature_std"].astype(np.float32)
    raw_latest = x[-1] * feature_std + feature_mean
    success_count = 0
    live_by_node = live.set_index("node_index")
    for node_index in range(raw_latest.shape[0]):
        if node_index not in live_by_node.index:
            continue
        row = live_by_node.loc[node_index]
        if not bool(row.get("success", False)):
            continue
        current_time = max(float(row["current_travel_time_s"]), 0.01)
        free_time = max(float(row["free_flow_travel_time_s"]), 0.01)
        raw_latest[node_index, 0] = float(row["current_speed_kph"])
        raw_latest[node_index, 1] = current_time
        raw_latest[node_index, 2] = current_time / free_time
        success_count += 1

    minute_of_day = local_hour * 60 + local_minute
    angle = 2.0 * math.pi * minute_of_day / (24.0 * 60.0)
    raw_latest[:, 6] = math.sin(angle)
    raw_latest[:, 7] = math.cos(angle)
    x[-1] = (raw_latest - feature_mean) / feature_std
    return x.astype(np.float32), raw_latest[:, 0].astype(np.float32), success_count


def display_label(raw_label: str) -> str:
    label = raw_label.replace("july_", "July ").replace("_", " ")
    return label.replace("am ", "AM ").replace("pm ", "PM ")
