"""EHR feature extraction (shared by physiology linkage and the fused model) + wearable/model features."""
import numpy as np, pandas as pd
from config import SNPS

COMPLICATIONS = {"Diabetic retinopathy", "Diabetic neuropathy"}
GL_CLASSES = {"metformin", "insulin (basal)", "insulin (bolus)", "SGLT2 inhibitor", "DPP-4 inhibitor", "sulfonylurea"}
SNAP_COLS = ["age", "sex_m", "bmi", "hba1c", "fasting_glucose", "ldl", "hdl", "triglycerides", "total_chol", "egfr", "creatinine", "alt"]

def _slope_per_year(df):
    if len(df) < 2: return 0.0
    t = (pd.to_datetime(df.date) - pd.to_datetime(df.date).min()).dt.days.values / 365.25
    return float(np.polyfit(t, df.value.values, 1)[0]) if t.max() > 0 else 0.0

def ehr_features(pat, labs, conds, meds, gen, asof=None):
    """Features for ONE patient. labs/conds/meds/gen are that patient's rows; pat is its patients.csv row (Series).
    asof: only labs dated strictly before this date are used (leakage guard)."""
    if asof is not None:
        labs = labs[pd.to_datetime(labs.date) < pd.Timestamp(asof).normalize()]
    f = {}
    for test, key in [("HbA1c", "hba1c"), ("fasting glucose", "fg"), ("LDL", "ldl"), ("eGFR", "egfr")]:
        s = labs[labs.test == test].sort_values("date")
        f[f"{key}_latest"] = float(s.value.iloc[-1]) if len(s) else np.nan
        f[f"{key}_slope_2y"] = _slope_per_year(s)
    ins = labs[labs.test == "fasting insulin"].sort_values("date"); g = labs[labs.test == "fasting glucose"].sort_values("date")
    f["homa_ir"] = float(ins.value.iloc[-1] * g.value.iloc[-1] / 405) if len(ins) and len(g) else np.nan
    active = conds[conds.status == "active"]
    f["comorbidity_count"] = sum(1 for c in active.condition if not c.startswith(("Prediabetes", "Type 2")))
    f["complication_flag"] = int(active.condition.isin(COMPLICATIONS).any() or active.condition.str.startswith("Chronic kidney").any())
    am = meds[meds.status == "active"]
    f["n_glucose_lowering_classes"] = len({("insulin" if c.startswith("insulin") else c) for c in am["class"] if c in GL_CLASSES})
    for flag, cls in [("on_metformin", "metformin"), ("on_sglt2", "SGLT2 inhibitor"), ("on_sulfonylurea", "sulfonylurea"),
                      ("on_dpp4", "DPP-4 inhibitor"), ("on_statin", "statin"), ("on_acearb", "ACE inhibitor / ARB")]:
        f[flag] = int((am["class"] == cls).any())
    f["on_insulin"] = int(am["class"].str.startswith("insulin").any())
    gmap = gen.set_index("snp_id").risk_allele_count.to_dict()
    for s in SNPS: f[f"snp_{s['snp_id']}"] = int(gmap.get(s["snp_id"], 0))
    f["genetic_risk_score"] = float(pat.genetic_risk_score)
    f["smoking_code"] = {"never": 0, "former": 1, "current": 2}[pat.smoking_status]
    f["family_history_diabetes"] = int(pat.family_history_diabetes); f["family_history_cvd"] = int(pat.family_history_cvd)
    f["years_since_dx"] = float(pat.years_since_dx)
    return f

def snapshot_features(pat):
    d = {c: float(pat[c]) for c in SNAP_COLS if c in pat.index}
    d["sex_m"] = float(pat.sex == "M")
    return d

def build_ehr_feature_table(tables, asof=None):
    pat, labs, conds, meds, gen = (tables[k] for k in ("patients", "labs", "conditions", "medications", "genomics"))
    L, C, M, G = (dict(tuple(x.groupby("patient_id"))) for x in (labs, conds, meds, gen))
    empty = lambda x: x.iloc[0:0]
    rows = {}
    for _, p in pat.iterrows():
        pid = p.patient_id
        f = ehr_features(p, L.get(pid, empty(labs)), C.get(pid, empty(conds)), M.get(pid, empty(meds)), G.get(pid, empty(gen)), asof)
        rows[pid] = {**snapshot_features(p), **f}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("patient_id")

SNAPSHOT_FEATS = SNAP_COLS
FULL_EXTRA = (["hba1c_slope_2y", "fg_slope_2y", "ldl_slope_2y", "egfr_slope_2y", "homa_ir", "comorbidity_count", "complication_flag",
               "n_glucose_lowering_classes", "on_metformin", "on_sglt2", "on_sulfonylurea", "on_dpp4", "on_insulin", "on_statin", "on_acearb",
               "genetic_risk_score", "smoking_code", "family_history_diabetes", "family_history_cvd", "years_since_dx"]
              + [f"snp_{s['snp_id']}" for s in SNPS])

# ---------------------------------------------------------------- wearable / model features
HORIZON = 12   # 12 x 5 min = 60 min ahead

def wearable_features(tl):
    """tl: fused timeline of ONE patient (5-min grid). Returns feature frame + target (glucose at t+60 min)."""
    tl = tl.sort_values("timestamp").reset_index(drop=True); g = tl.glucose
    X = pd.DataFrame({"patient_id": tl.patient_id, "timestamp": tl.timestamp})
    X["glucose"] = g
    for lag in (3, 6, 12): X[f"glucose_lag{lag*5}"] = g.shift(lag)
    X["glucose_roc15"] = g - g.shift(3); X["glucose_roc30"] = g - g.shift(6)
    X["glucose_mean60"] = g.rolling(12, min_periods=6).mean(); X["glucose_std60"] = g.rolling(12, min_periods=6).std()
    X["hr"] = tl.hr; X["hr_mean30"] = tl.hr.rolling(6, min_periods=1).mean()
    X["steps_30"] = tl.steps.rolling(6, min_periods=1).sum(); X["steps_60"] = tl.steps.rolling(12, min_periods=1).sum()
    X["active_energy_60"] = tl.active_energy.rolling(12, min_periods=1).sum()
    ts = pd.to_datetime(tl.timestamp); hod = ts.dt.hour + ts.dt.minute / 60
    X["tod_sin"] = np.sin(2 * np.pi * hod / 24); X["tod_cos"] = np.cos(2 * np.pi * hod / 24)
    X["asleep"] = (tl.sleep_stage >= 2).astype(int)
    # logged events (under-logged by design): carbs / bolus in last 3 h
    X["carbs_logged_3h"] = tl.carbs_logged.rolling(36, min_periods=1).sum(); X["bolus_logged_3h"] = tl.insulin_bolus_logged.rolling(36, min_periods=1).sum()
    mins_since = (~(tl.carbs_logged > 0)).astype(int).groupby((tl.carbs_logged > 0).cumsum()).cumsum() * 5
    X["min_since_logged_meal"] = mins_since.clip(upper=720)
    for c in ("sleep_hours_last_night", "sleep_efficiency_last_night", "rhr_trend_7d", "resting_hr", "hrv_rmssd_latest",
              "spo2_night_min", "wrist_temp_dev", "device_gap_minutes", "glucose_imputed", "hr_missing"):
        X[c] = tl[c]
    X["target"] = g.shift(-HORIZON)
    X["target_delta"] = X.target - X.glucose
    return X
