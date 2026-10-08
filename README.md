# Makati Road Segment Traffic Forecast

This Streamlit prototype runs the final A3T GCN traffic forecasting model for the Makati study route. It replaces the earlier SUMO and weather based prototype with live TomTom traffic input, the validated 88 node directed graph, and the final experiment checkpoints.

## Included models

The public interface uses the five checkpoint A3T GCN ensemble at 15, 30, and 60 minute horizons. The other final experiment checkpoints remain in the repository for reproducibility but are not exposed as website controls.

The network page contains both a geographic road map and a directed topology
graph. The map uses the TomTom segment centroids and the topology graph displays
the 88 nodes and 86 downstream road connections in their route order.

The A3T GCN ensemble averages checkpoints for training seeds 42, 52, 62, 72, and 82.

## Study configuration

| Item | Final configuration |
|---|---:|
| Graph nodes | 88 TomTom directional road segments |
| Directed road edges | 86 |
| History | 12 intervals at 15 minutes each |
| Features | 8 |
| Forecast horizons | 15, 30, and 60 minutes |
| Training partition | May |
| Validation partition | June |
| Test partition | July |

Normalization parameters were fitted on the May training partition only. The included performance tables come from the complete July test evaluation.

## Run locally in Visual Studio Code

Open the repository folder in Visual Studio Code, then run the following commands in PowerShell:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If a CUDA build of PyTorch is already installed in the active environment, Streamlit can use the RTX 3050. Inference is also fast enough to run on a CPU.

## Configure the TomTom API key

The live dashboard requires a TomTom API key:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Replace the placeholder with a current TomTom key.
3. Do not commit `.streamlit/secrets.toml`.

```toml
TOMTOM_API_KEY = "replace-with-your-key"
```

For Streamlit Community Cloud, add `TOMTOM_API_KEY` in the application's Secrets settings. The website fetches traffic automatically on the first visit and refreshes it every 15 minutes. A shared 15 minute cache prevents ordinary interface interactions from consuming another set of API calls.

## Validate the repository

```powershell
python -m unittest discover -s tests -v
python -m compileall app.py prototype_core.py tomtom_models.py
```

## Traffic condition thresholds

The map uses fixed forecast speed thresholds: heavy below 10 km/h, slow from 10 to below 20 km/h, moderate from 20 to below 30 km/h, and free flowing at 30 km/h or above.

The TomTom API supplies the newest traffic interval. Since the trained model requires a 12 interval input window, the preceding eleven intervals are retained from the prepared history during the initial live forecast.
