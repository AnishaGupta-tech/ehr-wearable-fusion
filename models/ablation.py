"""Ablation: does EHR help predict glucose 60 min ahead?  (patient-grouped CV, LightGBM on delta = g(t+60) - g(t))

  0 Persistence                       g(t+60) := g(t)
  1 Wearable-only
  2 Fused (snapshot EHR only)         + patients.csv latest values (demographics, BMI, latest labs)
  3 Fully fused                       + longitudinal slopes, HOMA-IR, comorbidity/complications, drug classes,
                                        per-SNP risk allele counts + polygenic score, smoking, family hx, years since dx
"""
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from config import data_dir
import features as F

WEARABLE_FEATS = ["glucose", "glucose_lag15", "glucose_lag30", "glucose_lag60", "glucose_roc15", "glucose_roc30", "glucose_mean60", "glucose_std60",
                  "hr", "hr_mean30", "steps_30", "steps_60", "active_energy_60", "tod_sin", "tod_cos", "asleep", "carbs_logged_3h", "bolus_logged_3h",
                  "min_since_logged_meal", "sleep_hours_last_night", "sleep_efficiency_last_night", "rhr_trend_7d", "resting_hr", "hrv_rmssd_latest",
                  "spo2_night_min", "wrist_temp_dev", "device_gap_minutes", "glucose_imputed", "hr_missing"]
MODELS = {"Wearable-only": WEARABLE_FEATS,
          "Fused (snapshot EHR only)": WEARABLE_FEATS + F.SNAPSHOT_FEATS,
          "Fully fused": WEARABLE_FEATS + F.SNAPSHOT_FEATS + F.FULL_EXTRA}
PARAMS = dict(objective="regression_l1", n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=100,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=8)

def make_dataset(profile, stride=3):
    fused = pd.read_parquet(data_dir(profile) / "fused_timeline.parquet")
    parts = []
    for pid, tl in fused.groupby("patient_id", sort=False):
        X = F.wearable_features(tl); ehr = tl.sort_values("timestamp").reset_index(drop=True)
        ehr_cols = [c for c in F.SNAPSHOT_FEATS + F.FULL_EXTRA if c in ehr.columns]
        X = pd.concat([X, ehr[ehr_cols]], axis=1); parts.append(X.iloc[::stride])
    d = pd.concat(parts).dropna(subset=["target", "glucose"]).reset_index(drop=True)
    return d

def run(profile="sample", stride=3, folds=5):
    d = make_dataset(profile, stride); groups = d.patient_id.values; y = d.target_delta.values
    preds = {"Persistence": d.glucose.values}
    imps = {}
    for name, cols in MODELS.items():
        oof = np.zeros(len(d))
        for tr, te in GroupKFold(folds).split(d, y, groups):
            m = lgb.LGBMRegressor(**PARAMS).fit(d.loc[tr, cols], y[tr]); oof[te] = d.glucose.values[te] + m.predict(d.loc[te, cols])
        preds[name] = oof
        m = lgb.LGBMRegressor(**PARAMS).fit(d[cols], y); imps[name] = dict(zip(cols, m.booster_.feature_importance("gain").tolist()))
    t = d.target.values; rows = []; per_pt = {}
    for name, p in preds.items():
        e = np.abs(p - t); per_pt[name] = pd.Series(e).groupby(groups).mean()
        row = dict(model=name, MAE=mean_absolute_error(t, p), RMSE=float(np.sqrt(np.mean((p - t) ** 2))), R2=r2_score(t, p),
                   AUC_hyper180=roc_auc_score(t > 180, p), AUC_hypo70=roc_auc_score(t < 70, -p) if (t < 70).any() and (t >= 70).any() else np.nan)
        rows.append(row)
    res = pd.DataFrame(rows).set_index("model")
    # per-patient paired tests
    tests = {}
    for a, b in [("Wearable-only", "Fused (snapshot EHR only)"), ("Fused (snapshot EHR only)", "Fully fused"), ("Wearable-only", "Fully fused")]:
        diff = per_pt[a] - per_pt[b]
        tests[f"{a} -> {b}"] = dict(mean_MAE_gain=float(diff.mean()), pct_patients_improved=float((diff > 0).mean() * 100),
                                    wilcoxon_p=float(wilcoxon(diff).pvalue) if len(diff) > 5 else None)
    out = data_dir(profile) / "results"; out.mkdir(exist_ok=True)
    res.round(4).to_csv(out / "ablation.csv")
    json.dump(dict(profile=profile, n_patients=int(d.patient_id.nunique()), n_rows=len(d), horizon_min=60, paired_tests=tests,
                   top_features={k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:15]) for k, v in imps.items()}), open(out / "ablation.json", "w"), indent=1)
    print(res.round(3).to_string()); print(json.dumps(tests, indent=1))
    return res

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "sample")
