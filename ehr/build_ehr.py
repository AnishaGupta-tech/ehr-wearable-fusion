"""Build multi-table EHR (Synthea-style) for a cohort.

Synthea supplies: demographics, diagnosis onset dates (T2D, prediabetes, hypertension, hyperlipidemia,
CAD), anthropometrics, lipids, smoking (never/former). Everything else (longitudinal labs, fasting insulin,
complications, medications, genotypes) is generated here with documented clinical correlations.
Genotypes and the polygenic score are SIMULATED.
"""
import numpy as np, pandas as pd
from config import *

def _read(name, **kw): return pd.read_csv(SYNTHEA_CSV / f"{name}.csv", **kw)

def ckd_epi_2021(cr, age, female):
    k = 0.7 if female else 0.9; a = -0.241 if female else -0.302
    return 142 * min(cr / k, 1) ** a * max(cr / k, 1) ** -1.2 * 0.9938 ** age * (1.012 if female else 1)

def creatinine_from_egfr(egfr, age, female):
    lo, hi = 0.3, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if ckd_epi_2021(mid, age, female) > egfr: lo = mid
        else: hi = mid
    return (lo + hi) / 2

def _latest(obs, desc_sub, before=None):
    s = obs[obs.DESCRIPTION.str.contains(desc_sub, regex=False, case=False)].copy()
    s["VALUE"] = pd.to_numeric(s.VALUE, errors="coerce"); s = s.dropna(subset=["VALUE"])
    return s.sort_values("DATE").groupby("PATIENT").VALUE.last()

def select_cohort(profile, rng):
    cfg = PROFILES[profile]; pts = _read("patients"); cond = _read("conditions")
    pts = pts[pts.DEATHDATE.isna()].copy()
    t2d = set(cond[cond.DESCRIPTION.str.contains("Diabetes mellitus type 2")].PATIENT)
    pre = set(cond[cond.DESCRIPTION.str.contains("Prediabetes")].PATIENT) - t2d
    pts["status"] = np.where(pts.Id.isin(t2d), "t2d", np.where(pts.Id.isin(pre), "prediabetes", "none"))
    out = []
    for st, frac in zip(["t2d", "prediabetes", "none"], cfg["mix"]):
        pool = pts[pts.status == st]; n = min(len(pool), round(cfg["n_patients"] * frac))
        out.append(pool.sample(n, random_state=int(rng.integers(1 << 31))))
    return pd.concat(out).reset_index(drop=True)

def build(profile="sample", seed=7):
    rng = np.random.default_rng(seed)
    coh = select_cohort(profile, rng); ids = set(coh.Id)
    cond = _read("conditions"); cond = cond[cond.PATIENT.isin(ids)]
    obs = _read("observations", usecols=["DATE", "PATIENT", "DESCRIPTION", "VALUE"]); obs = obs[obs.PATIENT.isin(ids)]
    h = _latest(obs, "Body Height"); w = _latest(obs, "Body Weight")
    ldl = _latest(obs, "LDL"); hdl = _latest(obs, "HDL"); tc = _latest(obs, "Cholesterol [Mass/volume]")
    tg = _latest(obs, "Triglyceride"); alt = _latest(obs, "Alanine"); a1c_syn = _latest(obs, "A1c")
    smk = obs[obs.DESCRIPTION.str.contains("Tobacco")].sort_values("DATE").groupby("PATIENT").VALUE.last()

    def onset(pid_conds, key):
        s = pid_conds[pid_conds.DESCRIPTION.str.contains(key, case=False)]
        return pd.to_datetime(s.START.min()) if len(s) else pd.NaT
    cond_by = {k: g for k, g in cond.groupby("PATIENT")}
    pat_rows, cond_rows, lab_rows, med_rows, gen_rows = [], [], [], [], []

    for _, p in coh.iterrows():
        pid = p.Id; st = p.status; female = p.GENDER == "F"
        age = int((ANCHOR - pd.Timestamp(p.BIRTHDATE)).days // 365.25)
        c = cond_by.get(pid, cond.iloc[0:0])
        # ---- anthropometrics
        hh = float(np.clip(h.get(pid, rng.normal(165 if female else 178, 7)), 140, 205))
        ww = float(np.clip(w.get(pid, rng.normal(75, 15)), 40, 190))
        if st != "none": ww *= rng.uniform(1.0, 1.15)  # Synthea under-weights cardiometabolic cohorts
        bmi = ww / (hh / 100) ** 2
        # ---- diabetes timeline
        if st == "t2d":
            dx = onset(c, "Diabetes mellitus type 2")
            if pd.isna(dx) or dx > ANCHOR - pd.Timedelta(days=60): dx = ANCHOR - pd.Timedelta(days=int(rng.integers(90, 365 * 12)))
        elif st == "prediabetes":
            dx = onset(c, "Prediabetes")
            if pd.isna(dx) or dx > ANCHOR: dx = ANCHOR - pd.Timedelta(days=int(rng.integers(60, 365 * 4)))
        else: dx = pd.NaT
        yrs_dx = (ANCHOR - dx).days / 365.25 if pd.notna(dx) else 0.0
        # ---- latest glycemic/metabolic values
        if st == "t2d":
            a1c = float(np.clip(a1c_syn.get(pid, 7.5) if a1c_syn.get(pid, 0) >= 6.0 else rng.normal(7.6, 1.3), 6.2, 12.5))
            a1c = float(np.clip(a1c + 0.08 * min(yrs_dx, 10) + rng.normal(0, 0.3), 6.2, 12.5))
        elif st == "prediabetes": a1c = float(rng.uniform(5.7, 6.4))
        else: a1c = float(np.clip(rng.normal(5.3, 0.25), 4.6, 5.6))
        fg = float(np.clip(28.7 * a1c - 46.7 + rng.normal(0, 9), 70, 330))
        homa_t = np.exp(rng.normal(0.35 + 0.045 * (bmi - 25) + 0.12 * (a1c - 5.5), 0.35)) if st != "none" else np.exp(rng.normal(0.1 + 0.04 * (bmi - 25), 0.3))
        homa_t = float(np.clip(homa_t, 0.5, 14)); fi = homa_t * 405 / fg
        ldl_v = float(np.clip(ldl.get(pid, rng.normal(110, 30)), 40, 230)); hdl_v = float(np.clip(hdl.get(pid, 50), 20, 95))
        tg_v = float(np.clip(tg.get(pid, 140), 40, 600)); tc_v = float(np.clip(ldl_v + hdl_v + tg_v / 5, 100, 330))
        alt_v = float(np.clip(alt.get(pid, 25) * (1 + 0.012 * (bmi - 25)), 8, 120))
        # ---- conditions
        crow = []
        def add(name, date, status="active"): crow.append((pid, name, pd.Timestamp(date).date(), status))
        if st == "t2d": add("Type 2 diabetes", dx)
        if st == "prediabetes": add("Prediabetes", dx)
        if st == "t2d":  # prediabetes preceded T2D
            pdx = onset(c, "Prediabetes")
            pdx = pdx if pd.notna(pdx) and pdx < dx else dx - pd.Timedelta(days=int(rng.integers(300, 365 * 5)))
            add("Prediabetes", pdx, "resolved")
        htn = onset(c, "hypertension"); hyl = onset(c, "Hyperlipidemia"); cad = onset(c, "Ischemic heart disease")
        if pd.isna(htn) and rng.random() < (0.45 if st == "t2d" else 0.2): htn = ANCHOR - pd.Timedelta(days=int(rng.integers(100, 3000)))
        dyslip = pd.notna(hyl) or ldl_v >= 160 or tg_v >= 200 or (st == "t2d" and rng.random() < 0.35)
        if dyslip and pd.isna(hyl): hyl = ANCHOR - pd.Timedelta(days=int(rng.integers(100, 2500)))
        if pd.notna(htn): add("Hypertension", min(htn, ANCHOR))
        if dyslip: add("Dyslipidemia", min(hyl, ANCHOR))
        if bmi >= 30: add("Obesity", ANCHOR - pd.Timedelta(days=int(rng.integers(200, 2500))))
        has_cad = pd.notna(cad) and cad < ANCHOR
        if has_cad: add("Coronary artery disease", cad)
        # complications ~ duration & HbA1c
        lp = -4.2 + 0.09 * yrs_dx + 0.45 * (a1c - 7) if st == "t2d" else -9
        pr = lambda off: 1 / (1 + np.exp(-(lp + off)))
        ret = st == "t2d" and rng.random() < pr(-0.4); neu = st == "t2d" and rng.random() < pr(0.1)
        ckd_logit = -2.6 + 0.07 * yrs_dx + 0.3 * (a1c - 7) + 0.03 * (age - 55) + (0.5 if pd.notna(htn) else 0) if st == "t2d" else -3.4 + 0.04 * (age - 55)
        ckd_stage = 0
        if rng.random() < 1 / (1 + np.exp(-ckd_logit)): ckd_stage = int(rng.choice([1, 2, 3], p=[0.2, 0.4, 0.4]))
        if st == "t2d":
            if ret: add("Diabetic retinopathy", dx + pd.Timedelta(days=int(rng.integers(365, 365 * 8))) if yrs_dx > 3 else dx + (ANCHOR - dx) * 0.7)
            if neu: add("Diabetic neuropathy", dx + (ANCHOR - dx) * rng.uniform(0.4, 0.95))
        egfr_t = {0: rng.normal(95, 10) - 0.25 * max(age - 50, 0), 1: rng.uniform(90, 118), 2: rng.uniform(60, 89), 3: rng.uniform(32, 59)}[ckd_stage]
        egfr_t = float(np.clip(egfr_t, 32, 125))
        if ckd_stage:
            add(f"Chronic kidney disease (stage {ckd_stage})", ANCHOR - pd.Timedelta(days=int(rng.integers(120, 1800))))
        # ---- medications
        meds = []
        def med(name, cls, dose, start=None, status="active"):
            start = start if start is not None else max(dx if pd.notna(dx) else ANCHOR - pd.Timedelta(days=700), ANCHOR - pd.Timedelta(days=int(rng.integers(60, 365 * 6))))
            meds.append((pid, name, cls, pd.Timestamp(start).date(), dose, status))
        on_insulin = on_sglt2 = on_su = on_dpp4 = on_met = False
        if st == "t2d":
            on_met = egfr_t >= 30 and rng.random() < 0.88
            if on_met: med("Metformin", "metformin", f"{rng.choice([500,1000,1500,2000])} mg/day")
            elif rng.random() < 0.3: med("Metformin", "metformin", "1000 mg/day", status="stopped")
            on_sglt2 = rng.random() < 0.18 + 0.15 * (ckd_stage > 0 or has_cad) + 0.1 * (a1c > 8)
            on_dpp4 = rng.random() < 0.2; on_su = rng.random() < 0.22 + 0.1 * (a1c > 8)
            on_insulin = rng.random() < 0.08 + 0.05 * min(yrs_dx, 10) + 0.12 * (a1c > 9)
            if on_sglt2: med("Empagliflozin", "SGLT2 inhibitor", "10 mg/day")
            if on_dpp4: med("Sitagliptin", "DPP-4 inhibitor", "100 mg/day")
            if on_su: med("Glimepiride", "sulfonylurea", f"{rng.choice([1,2,4])} mg/day")
            if on_insulin:
                med("Insulin glargine", "insulin (basal)", f"{rng.integers(10,40)} U/day")
                if a1c > 8 and rng.random() < 0.5: med("Insulin lispro", "insulin (bolus)", f"{rng.integers(4,10)} U/meal")
        elif st == "prediabetes" and rng.random() < 0.15:
            on_met = True; med("Metformin", "metformin", "500 mg/day")
        statin = dyslip or has_cad or (st == "t2d" and age >= 40 and rng.random() < 0.7)
        if statin: med("Atorvastatin", "statin", f"{rng.choice([10,20,40,80])} mg/day")
        if pd.notna(htn) or (st == "t2d" and ckd_stage) or has_cad:
            med(rng.choice(["Lisinopril", "Losartan"]), "ACE inhibitor / ARB", f"{rng.choice([10,20,40])} mg/day")
        # ---- genotypes (SIMULATED), enriched in cases per allele OR
        gscore, gen = 0.0, []
        for s in SNPS:
            f = s["freq"]
            if st != "none":
                orr = s["OR"] if st == "t2d" else 1 + (s["OR"] - 1) / 2
                f = f * orr / (1 - f + f * orr)
            n = int(rng.binomial(2, f)); gt = "".join(sorted([s["risk"]] * n + [s["other"]] * (2 - n)))
            gen.append((pid, s["snp_id"], s["gene"], gt, n, "SIMULATED")); gscore += np.log(s["OR"]) * n
        gmax = sum(2 * np.log(s["OR"]) for s in SNPS); gps = gscore / gmax
        # ---- longitudinal labs: walk backwards from the latest values
        nv = int(rng.integers(4, 9)); last_off = int(rng.integers(0, 30))
        extra = rng.choice(np.arange(last_off + 20, LAB_WINDOW_DAYS), nv - 1, replace=False)
        offs = sorted([last_off] + [int(x) for x in extra], reverse=True)   # days before anchor, oldest first
        dates = [ANCHOR - pd.Timedelta(days=o) for o in offs]
        drift_a1c = rng.normal(0.0, 0.35) if st == "t2d" else rng.normal(0.03, 0.12)   # change / year (latest minus earlier)
        egfr_slope = -rng.uniform(0.5, 3.5 if ckd_stage or st == "t2d" else 1.5)       # per year
        st_start = next((m[3] for m in meds if m[2] == "statin"), None)
        for j, d in enumerate(dates):
            k = (offs[j] - last_off) / 365.25   # years before latest visit
            a = float(np.clip(a1c - drift_a1c * k + (rng.normal(0, 0.12) if j < nv - 1 else 0), 4.4, 14))
            g = float(np.clip(fg + 28.7 * (a - a1c) + (rng.normal(0, 7) if j < nv - 1 else 0), 60, 360))
            ins = float(np.clip(fi * (g / fg) ** 0.5 * (np.exp(rng.normal(0, 0.1)) if j < nv - 1 else 1), 1.5, 90))
            lmult = (1.25 if (st_start and pd.Timestamp(st_start) > d) else 1) * np.exp(rng.normal(0, 0.05)) if j < nv - 1 else 1
            l = float(np.clip(ldl_v * lmult, 35, 260)); hd = float(np.clip(hdl_v * (np.exp(rng.normal(0, 0.04)) if j < nv - 1 else 1), 15, 100))
            tgx = float(np.clip(tg_v * (np.exp(rng.normal(0, 0.12)) if j < nv - 1 else 1), 35, 700)); tcx = l + hd + tgx / 5
            e = float(np.clip(egfr_t - egfr_slope * k + (rng.normal(0, 1.5) if j < nv - 1 else 0), 15, 130))
            if j == nv - 1: e = egfr_t
            cr = creatinine_from_egfr(e, age, female); at = float(np.clip(alt_v * (np.exp(rng.normal(0, 0.12)) if j < nv - 1 else 1), 6, 150))
            dd = d.date()
            for t, v, u in [("HbA1c", a, "%"), ("fasting glucose", g, "mg/dL"), ("fasting insulin", ins, "uIU/mL"),
                            ("total cholesterol", tcx, "mg/dL"), ("LDL", l, "mg/dL"), ("HDL", hd, "mg/dL"),
                            ("triglycerides", tgx, "mg/dL"), ("serum creatinine", cr, "mg/dL"), ("eGFR", e, "mL/min/1.73m2"), ("ALT", at, "U/L")]:
                lab_rows.append((pid, dd, t, round(v, 2), u))
            if j == nv - 1: final = dict(a1c=a, fg=g, fi=ins, tc=tcx, ldl=l, hdl=hd, tg=tgx, cr=cr, egfr=e, alt=at)
        fam_d = int(rng.random() < (0.55 if st == "t2d" else 0.35 if st == "prediabetes" else 0.2) + 0.1 * (gps > 0.5))
        fam_c = int(rng.random() < 0.3)
        sm = str(smk.get(pid, "Never"))
        sm = "former" if "Ex" in sm else "never"
        if rng.random() < 0.14: sm = "current"
        pat_rows.append(dict(patient_id=pid, age=age, sex="F" if female else "M", race=p.RACE, birth_date=p.BIRTHDATE,
            height_cm=round(hh, 1), weight_kg=round(ww, 1), bmi=round(ww / (hh / 100) ** 2, 1),
            hba1c=round(final["a1c"], 2), fasting_glucose=round(final["fg"], 1), fasting_insulin=round(final["fi"], 2),
            homa_ir=round(final["fg"] * final["fi"] / 405, 2), total_chol=round(final["tc"], 1), ldl=round(final["ldl"], 1),
            hdl=round(final["hdl"], 1), triglycerides=round(final["tg"], 1), creatinine=round(final["cr"], 2),
            egfr=round(final["egfr"], 1), alt=round(final["alt"], 1), diabetes_status=st,
            diabetes_onset_date=dx.date() if pd.notna(dx) else None, years_since_dx=round(yrs_dx, 2),
            smoking_status=sm, family_history_diabetes=fam_d, family_history_cvd=fam_c,
            genetic_risk_score=round(gps, 4), genetic_risk_flag=int(gps >= 0.5), genetics_label="SIMULATED"))
        cond_rows += crow; med_rows += meds; gen_rows += gen

    # genetic_risk_flag := top quartile of the cohort
    pat = pd.DataFrame(pat_rows); pat["genetic_risk_flag"] = (pat.genetic_risk_score >= pat.genetic_risk_score.quantile(0.75)).astype(int)
    out = data_dir(profile) / "ehr"; out.mkdir(exist_ok=True, parents=True)
    pat.to_csv(out / "patients.csv", index=False)
    pd.DataFrame(cond_rows, columns=["patient_id", "condition", "onset_date", "status"]).to_csv(out / "conditions.csv", index=False)
    pd.DataFrame(lab_rows, columns=["patient_id", "date", "test", "value", "unit"]).to_csv(out / "labs.csv", index=False)
    pd.DataFrame(med_rows, columns=["patient_id", "medication", "class", "start_date", "dose", "status"]).to_csv(out / "medications.csv", index=False)
    pd.DataFrame(gen_rows, columns=["patient_id", "snp_id", "gene", "genotype", "risk_allele_count", "label"]).to_csv(out / "genomics.csv", index=False)
    print(f"[ehr] {profile}: {len(pat)} patients -> {out}")
    return pat

if __name__ == "__main__":
    import sys; build(sys.argv[1] if len(sys.argv) > 1 else "sample")
