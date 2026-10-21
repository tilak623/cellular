# Comparative Performance Analysis of Time-Series and Machine Learning Models for Cellular Network Data Traffic Forecasting

Benchmarks **SARIMA** (statsmodels), **XGBoost** and an **LSTM** (PyTorch) on hourly macro-cell traffic (`M1`) from
`hetnet_traffic_hourly_1year.csv` (8,760 rows, 2024), with an interactive **Streamlit + Plotly** dashboard.

## Project structure

```
CellularForecastApp/
├── app.py                    # entry point: launches the Streamlit dashboard (reads results/)
├── data_loader.py            # loading, split, lag features, LSTM windows + scaling
├── train_models.py           # trains/evaluates all models, writes results + exports
├── requirements.txt          # ALL dependencies (training + dashboard)
├── requirements-deploy.txt   # lean dashboard-only deps (Render / cloud)
├── render.yaml               # Render blueprint
├── .streamlit/config.toml
├── data/hetnet_traffic_hourly_1year.csv
├── exports/                  # model weights (created by train_models.py)
└── results/                  # model_benchmark_results.csv, test_predictions.csv, 72 h plot, log, metadata
```

## Run locally

```bash
cd CellularForecastApp
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python data_loader.py                # optional: sanity-check the pipeline
python train_models.py               # full benchmark (SARIMA is the slowest step)
python app.py                        # launches http://localhost:8501  (same as: streamlit run app.py)
```

Useful flags: `--epochs 30`, `--skip-sarima`, `--skip-xgboost`, `--skip-lstm`, `--seed 7`,
`--sarima-order 1,0,1`, `--sarima-seasonal 1,1,1,24`, `--sarima-maxiter 30`, `--data-path path/to/file.csv`.

## Dashboard tabs

`Overview` (all models overlaid, live summary table) · `SARIMA` · `XGBoost` · `LSTM` (per-model KPIs, forecast, residuals,
predicted-vs-actual scatter, residual histogram, MAE by hour and weekday, hyperparameters) · `Benchmark` · `About`.
The sidebar sliders (horizon 24-168 h, window start) apply to every tab.

## Methodology

| Item | Setting |
|---|---|
| Split | Chronological 80 % / 20 % (7,008 train / 1,752 test hours), no shuffling |
| SARIMA | (1,0,1)x(1,1,1,24); fitted on train, then rolling 1-step-ahead filtering over test without refitting |
| XGBoost | `lag_1`, `lag_24`, `lag_168`, `hour_of_day`, `day_of_week`; early stopping on last 10 % of train |
| LSTM | 2 layers x 64 units, 24-hour windows, MinMaxScaler fitted on train only, early stopping |
| Metrics | RMSE, MAE (MB), training time (s), inference latency (ms/record, batch-amortised) |
| Post-processing | Predictions clipped at 0 MB |

All models are scored on identical test timestamps with one-step-ahead forecasts from true history, so the
comparison is like-for-like. Seasonal period 24 captures the daily cycle; the strong weekly pattern is only
captured by XGBoost (`lag_168`, `day_of_week`) - worth discussing in the paper. Run-to-run results vary slightly
(LSTM especially); report the seed and `results/run_metadata.json`.

## Deploy (public access)

Both platforms only need the dashboard, which reads `results/*.csv`. The full `requirements.txt` includes PyTorch,
which is too heavy for free cloud builds, so deploy from a branch that uses the lean dependency file.

**Prepare (once):**
```bash
python train_models.py                       # make sure results/ is populated
git init && git add . && git commit -m "Traffic forecasting benchmark"
git branch -M main
git remote add origin https://github.com/<you>/cellular-forecast-app.git
git push -u origin main

git checkout -b deploy
cp requirements-deploy.txt requirements.txt  # lean deps for the cloud
git commit -am "Use lean requirements for deployment"
git push -u origin deploy
git checkout main                            # keep working with the full requirements on main
```

### Streamlit Community Cloud
1. Open https://share.streamlit.io and sign in with GitHub, then **Create app**.
2. Repository: `<you>/cellular-forecast-app`, branch: `deploy`, main file path: `app.py`.
3. **Advanced settings** -> Python 3.11 -> **Deploy**. You get a public `https://<name>.streamlit.app` URL.

### Render
1. Open https://dashboard.render.com -> **New +** -> **Web Service** -> connect the repo, branch `deploy`.
2. Runtime: Python 3. Build command: `pip install -r requirements.txt`.
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.
4. Environment variable: `PYTHON_VERSION=3.11.9`. Pick the Free plan -> **Create Web Service**.
   (Alternatively **New + -> Blueprint** with `render.yaml` on the `main` branch; it installs `requirements-deploy.txt`.)

The free Render plan sleeps when idle, so the first request after a pause can take about a minute.

## Retraining after deployment
Re-run `python train_models.py` locally, commit the updated `results/` and `exports/` to the deployed branch, and push; the app redeploys automatically.
