# EHR × wearable fusion demo (synthetic)

Synthea EHR → EHR-conditioned simulated wearables (simglucose CGM + multi-device streams) → ingestion & alignment → fused patient timeline → ablation → Patient Twin dashboard.
All data is synthetic; genotypes, polygenic score and devices are **SIMULATED**. See [CONTRACT.md](CONTRACT.md) for schemas, the physiology linkage and the architecture diagram.

```bash
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
brew install openjdk@17 libomp                                   # Synthea needs Java 17; LightGBM needs libomp (macOS)
export PYTHONPATH=. PATH=/opt/homebrew/opt/openjdk@17/bin:$PATH

python run_pipeline.py --profile sample          # 50 patients x 7 days, a few minutes
python run_pipeline.py --profile full            # 800 patients x 14 days, ~1 hour on 10 cores
streamlit run dashboard/app.py                   # Patient Twin / Cohort / Results
pytest tests                                     # run on the sample profile first
```

Stages can be run separately: `--stages synthea,ehr,wearables,ingest,models`.

Layout: `ehr/` (EHR tables) · `wearables/` (param derivation + simulators) · `ingestion/` (raw → timeline → fused) · `features.py` · `models/ablation.py` · `dashboard/app.py` · `tests/`.
