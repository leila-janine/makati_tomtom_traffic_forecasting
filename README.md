# Makati Road Segment Traffic Forecast

This Streamlit prototype runs the final TomTom traffic forecasting models for the Makati study route. It replaces the earlier SUMO and weather based prototype with the validated 88 node directed graph and the final experiment checkpoints.

## Included models

The interface supports the following methods at 15, 30, and 60 minute horizons:

- Persistence baseline
- Shared node LSTM
- T GCN
- A3T GCN

The network page contains both a geographic road map and a directed topology
graph. The map uses the TomTom segment centroids and the topology graph displays
the 88 nodes and 86 downstream road connections in their route order.

The neural models include checkpoints for training seeds 42, 52, 62, 72, and 82. The application can run one checkpoint or average the predictions from all five seeds.

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

The observed July test mode does not require an API key. For the optional API assisted snapshot mode:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Replace the placeholder with a current TomTom key.
3. Do not commit `.streamlit/secrets.toml`.

```toml
TOMTOM_API_KEY = "replace-with-your-key"
```

For Streamlit Community Cloud, add `TOMTOM_API_KEY` in the application's Secrets settings.

## Validate the repository

```powershell
python -m unittest discover -s tests -v
python -m compileall app.py prototype_core.py tomtom_models.py
```

## Interpretation of live mode

The TomTom API returns a current snapshot rather than the complete 12 interval history required by the trained models. Live mode therefore replaces the newest interval and retains the preceding eleven intervals from the chosen July test history. It is suitable for demonstrating integration and inference. It is not an independently validated live forecast.

The traffic percentage panel reports the share of the 88 graph nodes assigned to each speed condition. It does not estimate the percentage of vehicles or traffic volume.
