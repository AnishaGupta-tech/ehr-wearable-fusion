"""Map a patient's EHR (features.ehr_features + patients.csv) to simulator parameters.

Physiology linkage (all multiplicative on a simglucose adult template):
  insulin sensitivity  <- HOMA-IR (primary), HbA1c, smoking, polygenic score, age
  glucose clearance    <- BMI (higher BMI -> slower clearance)
  insulin clearance    <- eGFR (lower eGFR -> slower insulin clearance -> prolonged insulin action, higher insulin levels)
  beta-cell response   <- HbA1c, diabetes duration, polygenic score, DPP-4 / sulfonylurea
  renal glucose excr.  <- SGLT2 inhibitor (lowers peaks)
  hypoglycaemia risk   <- sulfonylurea / insulin (secretion floor not suppressed at low glucose)
"""
import numpy as np, pandas as pd

def derive_params(pat, f, seed):
    rng = np.random.default_rng(seed)
    homa = f["homa_ir"]; a1c = f["hba1c_latest"]; bmi = float(pat.bmi); egfr = f["egfr_latest"]; gps = f["genetic_risk_score"]
    smoke = f["smoking_code"]; age = float(pat.age)
    sens = np.exp(-0.28 * np.log(max(homa, 0.4) / 1.5)) * np.exp(-0.10 * (a1c - 5.5)) \
           * {0: 1.0, 1: 0.97, 2: 0.92}[smoke] * (1 - 0.12 * gps) * (1 - 0.002 * max(age - 40, 0))
    sens = float(np.clip(sens, 0.25, 1.6))
    p = dict(sens=sens,
             k1_scale=float(1 / (1 + 0.025 * max(bmi - 22, 0))),                 # slower glucose clearance with BMI
             m4_scale=float(np.clip((egfr / 90) ** 0.7, 0.35, 1.2)),              # slower insulin clearance with low eGFR
             kp1_scale=float(0.92 if f["on_metformin"] else 1.0),                 # metformin: lower hepatic output
             ke1=0.0005 * (8 if f["on_sglt2"] else 1), ke2=339 * (0.62 if f["on_sglt2"] else 1.0),
             )
    beta = 1.0 if pat.diabetes_status == "none" else float(np.clip(0.85 - 0.075 * (a1c - 6) - 0.015 * f["years_since_dx"] - 0.18 * gps, 0.15, 0.95))
    if f["on_dpp4"]: beta = min(beta * 1.15, 1.0)
    p.update(beta=beta, sulfonylurea=bool(f["on_sulfonylurea"]), on_insulin=bool(f["on_insulin"]),
             gref=float(np.clip(f["fg_latest"], 70, 260)))
    # lifestyle priors (also EHR-linked): activity falls with BMI/age/diabetes burden
    steps_mean = float(np.clip(7600 - 90 * (bmi - 25) - 45 * max(age - 45, 0) - 150 * f["comorbidity_count"] + rng.normal(0, 900), 1500, 14000))
    p.update(steps_mean=steps_mean, cho_daily=float(np.clip(rng.normal(230, 40) + 1.2 * (bmi - 25), 120, 380)),
             rhr_base=float(np.clip(58 + 0.35 * (bmi - 22) + 0.1 * max(age - 40, 0) + 2.0 * (smoke == 2) + rng.normal(0, 3.5), 45, 95)),
             hrv_base=float(np.clip(48 - 0.45 * (age - 35) - 0.6 * (bmi - 24) - 1.5 * (a1c - 5.5) + rng.normal(0, 6), 12, 90)),
             spo2_base=float(np.clip(97.2 - 0.045 * max(bmi - 25, 0) + rng.normal(0, 0.4), 92.5, 99)),
             sleep_need=float(np.clip(rng.normal(7.1, 0.6), 5.5, 8.5)),
             meal_log_p=float(np.clip(rng.normal(0.72, 0.1) - 0.1 * (bmi >= 30), 0.35, 0.95)),
             insulin_log_p=float(np.clip(rng.normal(0.8, 0.08), 0.5, 0.97)))
    return p
