import numpy as np, pandas as pd, pytest
from config import data_dir, ANCHOR
from ingestion.align import _fill_small_gaps, fuse_with_ehr, load_raw_streams, align_to_timeline
from wearables.simulate import COLS
P = "sample"

@pytest.fixture(scope="module")
def ehr():
    d = data_dir(P) / "ehr"; return {k: pd.read_csv(d / f"{k}.csv") for k in ("patients", "labs", "conditions", "medications", "genomics")}

def test_bmi_consistent(ehr):
    p = ehr["patients"]; assert np.allclose(p.bmi, p.weight_kg / (p.height_cm / 100) ** 2, atol=0.1)

def test_latest_labs_match_patients(ehr):
    last = ehr["labs"].sort_values("date").groupby(["patient_id", "test"]).value.last().unstack(); p = ehr["patients"].set_index("patient_id")
    for lab, col, tol in [("HbA1c", "hba1c", 0.01), ("fasting glucose", "fasting_glucose", 0.1), ("LDL", "ldl", 0.1), ("eGFR", "egfr", 0.1)]:
        assert (last[lab] - p[col]).abs().max() <= tol, lab

def test_homa_ir_derived(ehr):
    last = ehr["labs"].sort_values("date").groupby(["patient_id", "test"]).value.last().unstack(); p = ehr["patients"].set_index("patient_id")
    calc = (last["fasting glucose"] * last["fasting insulin"] / 405).reindex(p.index); assert (p.homa_ir - calc).abs().max() < 0.05

def test_lab_visits_4_to_8_within_2y(ehr):
    n = ehr["labs"].groupby("patient_id").date.nunique(); assert n.between(4, 8).all()
    assert (pd.to_datetime(ehr["labs"].date) >= ANCHOR - pd.Timedelta(days=731)).all()

def test_meds_match_conditions(ehr):
    c = ehr["conditions"]; m = ehr["medications"]
    insulin_pts = set(m[m["class"].str.startswith("insulin")].patient_id); t2d = set(c[c.condition == "Type 2 diabetes"].patient_id)
    assert insulin_pts <= t2d
    htn_or_ckd = set(c[c.condition.isin(["Hypertension"]) | c.condition.str.startswith("Chronic")].patient_id)
    arb = set(m[m["class"] == "ACE inhibitor / ARB"].patient_id)
    assert all(p in htn_or_ckd or p in set(c[c.condition == "Coronary artery disease"].patient_id) for p in arb)

def test_complications_need_diabetes(ehr):
    c = ehr["conditions"]; t2d = set(c[c.condition == "Type 2 diabetes"].patient_id)
    assert set(c[c.condition.str.startswith("Diabetic")].patient_id) <= t2d

def test_genomics_complete_and_labeled(ehr):
    g = ehr["genomics"]; assert g.groupby("patient_id").size().eq(5).all() and (g.label == "SIMULATED").all() and g.risk_allele_count.between(0, 2).all()
    assert ehr["patients"].genetic_risk_score.between(0, 1).all()

def test_raw_schema_and_native_rates(ehr):
    r = load_raw_streams(ehr["patients"].patient_id[0], P); assert list(r.columns) == COLS
    cgm = r[r.metric == "glucose"].start_time.diff().dropna(); assert (cgm >= pd.Timedelta(minutes=5)).all()
    hr = r[r.metric == "heart_rate"].start_time.diff().dropna().dt.total_seconds() / 60; assert hr.nunique() > 3   # irregular

def test_small_gaps_only():
    s = pd.Series([100, np.nan, np.nan, 130, np.nan, np.nan, np.nan, np.nan, 170.0]); out, imp = _fill_small_gaps(s, 3)
    assert imp[[1, 2]].all() and not imp[[4, 5, 6, 7]].any() and out[[4, 5, 6, 7]].isna().all()

def test_fuse_no_future_lab_leakage(ehr):
    pid = ehr["patients"].patient_id[0]; tl = align_to_timeline(load_raw_streams(pid, P)).iloc[:300]
    t = {k: v.copy() for k, v in ehr.items()}
    future = pd.DataFrame([dict(patient_id=pid, date=(tl.timestamp.iloc[100]).date().isoformat(), test="HbA1c", value=99.0, unit="%")])
    t["labs"] = pd.concat([t["labs"], future], ignore_index=True)
    f = fuse_with_ehr(tl, t)
    day_of_lab = pd.Timestamp(future.date[0]); before = f[f.timestamp < day_of_lab + pd.Timedelta(days=1)]
    assert (before.hba1c_latest < 50).all()           # lab dated D is invisible on day D (strictly before)
    assert (f[f.timestamp >= day_of_lab + pd.Timedelta(days=1)].hba1c_latest == 99.0).all()

# ---------------------------------------------------------------- adverse-event labels
def _tl(g):
    n = len(g); z = np.zeros(n)
    return pd.DataFrame(dict(patient_id="p", timestamp=pd.date_range("2026-10-01", periods=n, freq="5min"), glucose=g, hr=z + 70, steps=z, active_energy=z,
                             sleep_stage=z, carbs_logged=z, insulin_bolus_logged=z, sleep_hours_last_night=z, sleep_efficiency_last_night=z, rhr_trend_7d=z,
                             resting_hr=z, hrv_rmssd_latest=z, spo2_night_min=z, wrist_temp_dev=z, device_gap_minutes=z, glucose_imputed=z, hr_missing=z))

def test_event_label_uses_only_future_window():
    from models.events import patient_frame, WINDOW
    g = np.full(80, 120.0); g[50] = 200.0                      # single spike at index 50
    X = patient_frame(_tl(g))
    assert X.y_hyper[50 - WINDOW:50].eq(1).all()               # 2 h before the spike -> positive
    assert X.y_hyper[50] == 0 and X.y_hyper[51] == 0           # spike itself / after it is not "future"
    assert X.y_hyper[:50 - WINDOW].eq(0).all()
    assert X.y_hyper[-WINDOW:].isna().all()                    # incomplete look-ahead window is dropped

def test_onsets_and_lead_time():
    from models.events import find_onsets, lead_times
    g = np.full(60, 120.0); g[40:] = 200.0; g[10:12] = 175.0
    on = find_onsets(_tl(g)); assert [(k, t) for t, k in on] == [("hyper", pd.Timestamp("2026-10-01") + pd.Timedelta(minutes=5 * 40))]
    d = _tl(g)[["patient_id", "timestamp"]]; flag = np.zeros(60, bool); flag[34] = True            # alert 30 min before onset
    n, leads = lead_times(d, flag, pd.DataFrame(on, columns=["timestamp", "kind"]).assign(patient_id="p"), "hyper")
    assert n == 1 and leads == [30.0]

# ---------------------------------------------------------------- streaming
def test_stream_features_equal_batch_and_scores_match_oof():
    from models.events import patient_frame, FUSED, HISTORY_ROWS, score_stream
    import json
    fused = pd.read_parquet(data_dir(P) / "fused_timeline.parquet"); pid = fused.patient_id.iloc[0]
    tl = fused[fused.patient_id == pid].sort_values("timestamp").reset_index(drop=True); full = patient_frame(tl)
    meta = json.load(open(data_dir(P) / "results" / "events.json")); risk = pd.read_parquet(data_dir(P) / "results" / "event_risk.parquet"); risk = risk[risk.patient_id == pid].set_index("timestamp")
    checked = 0
    for i in (400, 777, 1200, 1500):
        part = patient_frame(tl.iloc[:i + 1].tail(HISTORY_ROWS)).iloc[-1]       # only data up to i, as in a live feed
        np.testing.assert_allclose(part[FUSED].astype(float).values, full.loc[i, FUSED].astype(float).values, rtol=1e-5, equal_nan=True)
        sc = score_stream(P, tl.iloc[:i + 1], meta["patient_fold"][pid]); t = tl.timestamp[i]
        if sc["in_range"] and t in risk.index: assert abs(sc["hyper"][0] - risk.loc[t, "p_hyper"]) < 1e-4; checked += 1
    assert checked >= 1
