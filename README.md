# EHR × wearable fusion demo (synthetic)

## Summary (for reviewers)

**Problem and why India.** India has an estimated 101 million people with diabetes and 136 million with prediabetes (ICMR-INDIAB, 2023), and South Asians develop type 2 diabetes at lower BMI and younger ages than most populations. Most are seen a few times a year, so a dangerous glucose swing between visits is invisible to the clinician. Continuous glucose monitors and smartwatches are spreading but are not linked to the health record, and a doctor cannot read every trace. We ask: **can fusing a patient's record (labs, diagnoses, medicines, genetics) with their wearable streams warn about a high (> 180 mg/dL) or low (< 70) glucose event up to 2 hours ahead, say why, and let a clinician see which patients need attention first?**

**What it does (a type 2 diabetes digital twin).**
* **Fused twin:** 800 synthetic patients (Synthea EHR + simulated labs, medicines, SNPs, polygenic score) → EHR-conditioned simulation of Dexcom-style CGM and Apple-Watch-style streams at native rates → ingestion and alignment to one 5-minute timeline. EHR values are joined only from labs dated *before* each timestamp (leakage test). No real patient data; MIMIC-IV is not used.
* **Early warning:** LightGBM classifiers give P(glucose > 180) and P(< 70) within 2 h, plus 1 h and 2 h glucose forecasts with an 80% band (conformalised on held-out patients). An alert needs risk above threshold now and 10 min earlier. Three alert sensitivities (sensitive / balanced / precise) let a clinic trade missed events against false alarms.
* **Doctor interface (Streamlit):** *Triage board* ranks the whole panel by current risk; *Patient view* shows risk, alert flag, forecast bands, a "why flagged" SHAP chart (tagged wearable vs health record) and a what-if panel (toggle a medicine or activity and see the risk move); *Live monitor* streams a patient's data tick by tick, re-scores only with data received so far, and keeps an alert log that marks each alert confirmed or false once 2 h have passed; *Model results* shows every metric below.
* **Real-time path:** `models/events.py::score_stream` rebuilds features from the received history and scores with the held-out-fold model. A test checks that streamed features and risks equal the batch ones, so there is no look-ahead.

**Results (800 patients × 14 days, patient-grouped 5-fold CV, 1.2 M test moments from held-out patients).** An event is glucose out of range for at least 15 minutes (3 readings), the consensus CGM definition, which ignores single-sample sensor spikes. Hypoglycaemia is rare (2.6% of in-range moments), so precision is judged against that base rate.

| Event | Model | AUC | Avg. precision | Alert-episode precision | Events caught | Median warning | Alerts / patient-day |
|---|---|---|---|---|---|---|---|
| > 180 | Trend rule (baseline) | 0.748 | 0.31 | 0.30 | 93.7% | 40 min | 3.9 |
| > 180 | Wearable-only | 0.932 | 0.66 | 0.49 | 89.7% | 80 min | 1.9 |
| > 180 | **Fused** | **0.932** | **0.66** | **0.49** | 89.1% | 80 min | 1.8 |
| < 70 | Trend rule (baseline) | 0.843 | 0.16 | 0.19 | 77.0% | 40 min | 1.1 |
| < 70 | Wearable-only | 0.957 | 0.47 | 0.37 | 70.7% | 70 min | 0.50 |
| < 70 | **Fused** | **0.956** | **0.46** | **0.37** | 69.7% | 65 min | 0.49 |

("Balanced" alert setting. "Alert-episode precision" = share of alert episodes followed by an event within 2 h; "events caught" = an alert fired in the 2 h before glucose first left range after an hour in range.) For low glucose, average precision is 18 times the base rate, and richer physiology features (insulin and carbohydrate on board, smoothed trend and slope, the patient's own low-glucose history over the last 24 and 72 h, time since last low, the same hour on previous days) lifted it from 0.32 to 0.47 on the same labels.

Alert sensitivity for the fused model:

| Event | Setting | Events caught | Alert-episode precision | Row-level precision | Alerts / patient-day |
|---|---|---|---|---|---|
| > 180 | sensitive | 97.1% | 0.32 | 0.44 | 2.4 |
| > 180 | balanced | 89.1% | 0.49 | 0.59 | 1.8 |
| > 180 | precise | 85.1% | 0.54 | 0.63 | 1.7 |
| < 70 | sensitive | 95.3% | 0.13 | 0.21 | 1.3 |
| < 70 | balanced | 69.7% | 0.37 | 0.49 | 0.49 |
| < 70 | precise | 49.3% | 0.55 | 0.66 | 0.23 |

Glucose forecast error (fused, median): 11.9 mg/dL at +1 h (persistence 17.4) and 16.2 at +2 h (persistence 26.4); the 80% band covers 80% of held-out outcomes.

**Does the health record help?** Once a patient has a day of CGM, no: the personal-history features carry what the record knew, and the fused and wearable-only models now score the same overall. In the first 24 hours, before that history exists, the record helps for high glucose (AUC 0.920 vs 0.910, average precision 0.64 vs 0.60) and does not for low glucose (0.39 vs 0.41, within noise). Against a wearable model without personal-history features, the record lowers 60-minute glucose error from 13.5 to 12.8 mg/dL (`models/ablation.py`). So the record's value here is a day-one prior, and the simulator's EHR-to-physiology link is what it recovers.

**Impact.** With the balanced setting a monitored patient triggers about 2.3 alerts a day (1.8 high, 0.5 low), of which about 5 in 10 (high) and 4 in 10 (low) are followed by a real event. That is too many to send to a doctor, so the design is tiered: the patient is alerted first, and the **Triage board** shows the clinician only who is in alert at the moment, so a small team can supervise a large panel between visits. A clinic that wants fewer interruptions picks the precise setting (0.23 low-glucose alerts a day, 55% of them real, at the cost of missing about half of low events).

**Limitations.**
* The simulator is driven by the EHR, so the fusion gain shows the pipeline can recover that link, not that it exists in real patients. Validation on real, consented data is the necessary next step.
* Hypoglycaemia is the weaker result: about 6 in 10 low-glucose alert episodes are false at the balanced setting, and the precise setting misses half of the events. Model tuning (depth, regularisation, more training rows, shorter horizons) did not move it; what is left looks like unlogged insulin and meals, which no feature can recover.
* Thresholds are chosen on pooled held-out scores, so they are mildly optimistic, and they are not tuned to a clinical cost of missed versus false alarms.
* What-if answers come from associations the model learned on synthetic data; they are not causal treatment advice.
* The live monitor replays simulated streams; connecting a real device feed needs only an adapter that appends rows to the timeline.
* Genotypes, polygenic score and devices are simulated. This is research code, not a medical device.

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

Stages can be run separately: `--stages synthea,ehr,wearables,ingest,models,events` (`events` = adverse-event model, ~10 min on `full`).

Layout: `ehr/` (EHR tables) · `wearables/` (param derivation + simulators) · `ingestion/` (raw → timeline → fused) · `features.py` · `models/ablation.py` · `models/events.py` (adverse-event alerts) · `dashboard/app.py` · `tests/`.
