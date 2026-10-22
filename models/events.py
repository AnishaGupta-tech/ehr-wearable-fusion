"""Adverse-event model: will glucose leave 70-180 mg/dL within the next 2 hours?

Population: timestamps where glucose is currently in range (we predict NEW events, not ones already happening).
Labels:     hyper = glucose > 180 for >= 15 min (3 readings) within t+5..t+120 min ;  hypo = < 70 for >= 15 min (consensus CGM event definition).
Models:     LightGBM classifiers, patient-grouped CV.  Wearable-only vs fused (wearables + full EHR) vs a trend rule.
Also fits 10/50/90% quantile regressors for glucose at t+60 min (forecast band) and keeps the per-fold fused
models so the dashboard can explain an alert (SHAP contributions).
"""
import json, sys, numpy as np, scipy.signal, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from config import data_dir
import features as F
from models.ablation import WEARABLE_FEATS, MODELS

HIGH, LOW = 180.0, 70.0
WINDOW = 24            # 24 x 5 min = 2 h
LEAD_MAX_MIN = 120
SUSTAIN = 3            # readings (15 min) out of range that make an event
QUIET = 12             # onset = out of range after 1 h fully in range
EXTRA_WEARABLE = ["glucose_min60", "glucose_max60", "glucose_min180", "glucose_max180", "glucose_roc5", "glucose_roc60", "glucose_accel",
                  "min_since_logged_bolus", "steps_120"]
# insulin/carb exposure, effort and the patient's own recent low-glucose history (all causal)
EXTRA_PHYS = ["iob_logged", "cob_logged", "iob_per_cob", "active_energy_120", "hr_excess_60", "n_low_24h", "min_glucose_24h", "frac_below_90_24h",
              "glucose_cv_24h", "min_since_low", "n_low_72h", "mean_glucose_24h",
              "glucose_ewm15", "glucose_ewm30", "slope_ewm30", "slope_30", "slope_60", "min_to_70_at_slope", "min_since_peak3h", "n_low_same_tod_3d",
              # high-glucose counterparts of the personal-history features above
              "n_high_24h", "max_glucose_24h", "frac_above_160_24h", "n_high_same_tod_3d", "min_to_180_at_slope", "glucose_minus_mean24h", "glucose_extrap60"]
EXTRA_WEARABLE = EXTRA_WEARABLE + EXTRA_PHYS
WEARABLE = WEARABLE_FEATS + EXTRA_WEARABLE
FUSED = MODELS["Fully fused"] + EXTRA_WEARABLE
FEATURE_SETS = {"Wearable-only": WEARABLE, "Fused": FUSED}
CONFIRM_MIN = 10       # an alert needs risk above threshold now AND 10 min earlier (suppresses single-sample blips)
EVENTS = ("hyper", "hypo")
# regularised: EHR columns are constant per patient, so looser trees memorise training patients (fused AP -0.03 on held-out patients)
CLF = dict(objective="binary", n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=300,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=10.0, verbose=-1, n_jobs=8)
REG = dict(objective="quantile", n_estimators=250, learning_rate=0.05, num_leaves=31, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=8)
QUANTILES = (0.1, 0.5, 0.9)

def patient_frame(tl):
    """Features + event labels for ONE patient's fused timeline. Features at t use data <= t; labels use t+5..t+120 only."""
    tl = tl.sort_values("timestamp").reset_index(drop=True)
    X = F.wearable_features(tl)
    X = pd.concat([X, tl[[c for c in F.SNAPSHOT_FEATS + F.FULL_EXTRA if c in tl.columns]]], axis=1)
    g = tl.glucose; n = len(tl)
    X["glucose_min60"] = g.rolling(12, min_periods=6).min(); X["glucose_max60"] = g.rolling(12, min_periods=6).max()
    X["glucose_min180"] = g.rolling(36, min_periods=12).min(); X["glucose_max180"] = g.rolling(36, min_periods=12).max()
    X["glucose_roc5"] = g - g.shift(1); X["glucose_roc60"] = g - g.shift(12); X["glucose_accel"] = X.glucose_roc15 - X.glucose_roc15.shift(3)
    bol = tl.insulin_bolus_logged > 0
    X["min_since_logged_bolus"] = ((~bol).astype(int).groupby(bol.cumsum()).cumsum() * 5).clip(upper=720); X["steps_120"] = tl.steps.rolling(24, min_periods=1).sum()
    a_ins, a_cho = 0.5 ** (1 / 48), 0.5 ** (1 / 12)                      # half-lives 4 h (insulin) and 1 h (carbs)
    X["iob_logged"] = scipy.signal.lfilter([1], [1, -a_ins], tl.insulin_bolus_logged.fillna(0).values)
    X["cob_logged"] = scipy.signal.lfilter([1], [1, -a_cho], tl.carbs_logged.fillna(0).values)
    X["iob_per_cob"] = X.iob_logged / (X.cob_logged / 10 + 1)
    X["active_energy_120"] = tl.active_energy.rolling(24, min_periods=1).sum()
    X["hr_excess_60"] = tl.hr.rolling(12, min_periods=1).mean() - tl.resting_hr
    low = (g < LOW).astype(float).where(g.notna())
    X["n_low_24h"] = low.rolling(288, min_periods=48).sum(); X["n_low_72h"] = low.rolling(864, min_periods=144).sum()
    X["min_glucose_24h"] = g.rolling(288, min_periods=48).min(); X["mean_glucose_24h"] = g.rolling(288, min_periods=48).mean()
    X["frac_below_90_24h"] = (g < 90).astype(float).where(g.notna()).rolling(288, min_periods=48).mean()
    X["glucose_cv_24h"] = g.rolling(288, min_periods=48).std() / X.mean_glucose_24h
    islow = low.fillna(0) > 0
    X["min_since_low"] = ((~islow).astype(int).groupby(islow.cumsum()).cumsum() * 5).clip(upper=1440)
    e3, e6 = g.ewm(halflife=3, ignore_na=True).mean(), g.ewm(halflife=6, ignore_na=True).mean()      # sensor-noise smoothing
    X["glucose_ewm15"], X["glucose_ewm30"] = e3, e6; X["slope_ewm30"] = (e3 - e3.shift(6)) / 6
    for w_, nm_ in ((6, "slope_30"), (12, "slope_60")):                                    # least-squares slope (mg/dL per bin) over the last w_ bins
        x_ = np.arange(w_) - (w_ - 1) / 2; sl = np.full(len(g), np.nan); sl[w_ - 1:] = np.convolve(g.values.astype(float), x_[::-1], "valid") / (x_ ** 2).sum(); X[nm_] = sl
    X["min_to_70_at_slope"] = np.where(X.slope_30 < -0.2, ((g - LOW) / -X.slope_30 * 5).clip(upper=600), 600.0)
    arr = np.nan_to_num(g.values, nan=-1.0); pad = np.r_[np.full(35, -1.0), arr]
    X["min_since_peak3h"] = (35 - np.lib.stride_tricks.sliding_window_view(pad, 36).argmax(axis=1)) * 5
    lw = low.rolling(49, center=True, min_periods=25).max(); X["n_low_same_tod_3d"] = sum(lw.shift(288 * k_).fillna(0) for k_ in (1, 2, 3))
    high = (g > HIGH).astype(float).where(g.notna())
    X["n_high_24h"] = high.rolling(288, min_periods=48).sum(); X["max_glucose_24h"] = g.rolling(288, min_periods=48).max()
    X["frac_above_160_24h"] = (g > 160).astype(float).where(g.notna()).rolling(288, min_periods=48).mean()
    hw = high.rolling(49, center=True, min_periods=25).max(); X["n_high_same_tod_3d"] = sum(hw.shift(288 * k_).fillna(0) for k_ in (1, 2, 3))
    X["min_to_180_at_slope"] = np.where(X.slope_30 > 0.2, ((HIGH - g) / X.slope_30 * 5).clip(upper=600), 600.0)
    X["glucose_minus_mean24h"] = g - X.mean_glucose_24h; X["glucose_extrap60"] = e3 + 12 * X.slope_ewm30
    # consensus CGM event definition: >= 15 min (3 consecutive readings) out of range, which ignores single-sample sensor spikes
    hi3, lo3 = g.rolling(SUSTAIN).min(), g.rolling(SUSTAIN).max()           # all SUSTAIN readings ending here are > HIGH / < LOW
    span = WINDOW - SUSTAIN + 1
    fmax = hi3.iloc[::-1].rolling(span, min_periods=span // 2).max().iloc[::-1].shift(-SUSTAIN)
    fmin = lo3.iloc[::-1].rolling(span, min_periods=span // 2).min().iloc[::-1].shift(-SUSTAIN)
    X["y_hyper"] = (fmax > HIGH).astype(float).where(fmax.notna()); X["y_hypo"] = (fmin < LOW).astype(float).where(fmin.notna())
    X.loc[max(n - WINDOW, 0):, ["y_hyper", "y_hypo"]] = np.nan          # incomplete look-ahead window
    X["target120"] = g.shift(-WINDOW); X["target120_delta"] = X.target120 - g          # 2 h ahead
    X["in_range"] = ((g >= LOW) & (g <= HIGH)).values
    X["idx"] = np.arange(n)
    return X

def find_onsets(tl):
    """Event onsets: first out-of-range sample after >= 1 h continuously in range. Returns (timestamps, kinds)."""
    tl = tl.sort_values("timestamp").reset_index(drop=True); g = tl.glucose
    inr = ((g >= LOW) & (g <= HIGH)).astype(float)
    quiet = inr.shift(1).rolling(QUIET).sum() == QUIET
    hy = quiet & (g.rolling(SUSTAIN).min().shift(-(SUSTAIN - 1)) > HIGH); ho = quiet & (g.rolling(SUSTAIN).max().shift(-(SUSTAIN - 1)) < LOW)
    return [(tl.timestamp[i], "hyper") for i in np.flatnonzero(hy)] + [(tl.timestamp[i], "hypo") for i in np.flatnonzero(ho)]

def make_dataset(profile, eval_stride=1, train_stride=3):
    fused = pd.read_parquet(data_dir(profile) / "fused_timeline.parquet")
    parts, ons, days = [], [], {}
    for pid, tl in fused.groupby("patient_id", sort=False):
        X = patient_frame(tl); X["train"] = (X.idx % train_stride == 0)
        parts.append(X[X.idx % eval_stride == 0])
        ons += [dict(patient_id=pid, timestamp=t, kind=k) for t, k in find_onsets(tl)]
        days[pid] = (tl.timestamp.max() - tl.timestamp.min()).total_seconds() / 86400
    d = pd.concat(parts, ignore_index=True)
    num = d.select_dtypes("float64").columns; d[num] = d[num].astype("float32")
    return d, pd.DataFrame(ons, columns=["patient_id", "timestamp", "kind"]), days

def best_f1_threshold(y, s):
    p, r, thr = precision_recall_curve(y, s); f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f1)); return float(thr[i]), float(p[i]), float(r[i])

OPS = ("sensitive", "balanced", "precise")
def op_thresholds(y, s):
    """Three alert operating points: sensitive (recall >= 85%), balanced (max F1), precise (precision >= 60%, else best precision at recall >= 5%)."""
    p, r, thr = precision_recall_curve(y, s); p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12); sens = np.flatnonzero(r >= 0.85); prec = np.flatnonzero((p >= 0.6) & (r >= 0.05))
    i = dict(balanced=int(np.argmax(f1)), sensitive=int(sens[-1]) if len(sens) else 0,
             precise=int(prec[0]) if len(prec) else int(np.argmax(np.where(r >= 0.05, p, -1))))
    return {k: (float(thr[v]), float(p[v]), float(r[v])) for k, v in i.items()}

def alert_episodes(ts, flag, gap_min=30):
    """Count alert episodes: consecutive alerting rows, merged when closer than gap_min."""
    a = np.sort(ts[flag]).astype("datetime64[m]").astype(np.int64)
    return 0 if len(a) == 0 else 1 + int((np.diff(a) > gap_min).sum())

def alert_starts(ts, flag, gap_min=30):
    """Positions (into ts/flag) of the first row of each alert episode."""
    pos = np.flatnonzero(flag)
    if len(pos) == 0: return pos
    t = ts[pos].astype("datetime64[m]").astype(np.int64); o = np.argsort(t); pos, t = pos[o], t[o]
    return pos[np.r_[True, np.diff(t) > gap_min]]

def lead_times(d, flag, onsets, kind):
    """Per onset of `kind`: earliest alert in the 2 h before onset. Returns (n_onsets, lead_minutes list)."""
    leads = []; on = onsets[onsets.kind == kind]; byp = {k: v for k, v in d.assign(_f=flag).groupby("patient_id")}
    for r in on.itertuples():
        g = byp.get(r.patient_id)
        if g is None: continue
        ts = g.timestamp.values; T = np.datetime64(r.timestamp)
        m = g._f.values & (ts >= T - np.timedelta64(LEAD_MAX_MIN, "m")) & (ts < T)
        if m.any(): leads.append(float((T - ts[m].min()) / np.timedelta64(1, "m")))
    return len(on), leads

def run(profile="sample", eval_stride=1, train_stride=3, folds=5):
    d, onsets, days = make_dataset(profile, eval_stride, train_stride)
    fold = np.full(len(d), -1)
    for k, (_, te) in enumerate(GroupKFold(folds).split(d, groups=d.patient_id)): fold[te] = k
    patient_fold = d.assign(f=fold).groupby("patient_id").f.first().astype(int).to_dict()
    out = data_dir(profile) / "results"; (out / "event_models").mkdir(parents=True, exist_ok=True)
    pop = d.in_range.values & d.y_hyper.notna().values
    score = {}                                                     # (model, event) -> OOF score array (NaN outside population)
    for ev in EVENTS:
        y = d[f"y_{ev}"].values
        for name, cols in FEATURE_SETS.items():
            oof = np.full(len(d), np.nan)
            for k in range(folds):
                tr = (fold != k) & pop & d.train.values; te = (fold == k) & pop
                m = lgb.LGBMClassifier(**CLF).fit(d.loc[tr, cols], y[tr]); oof[te] = m.predict_proba(d.loc[te, cols])[:, 1]
                if name == "Fused": m.booster_.save_model(str(out / "event_models" / f"{ev}_fold{k}.txt"))
            score[(name, ev)] = oof; print(f"[events] {ev} / {name} done", flush=True)
        trend = (d.glucose + 2 * d.glucose_roc30.fillna(0)).values.astype(float)
        score[("Trend rule", ev)] = np.where(pop, trend if ev == "hyper" else -trend, np.nan)

    tsv = d.timestamp.values; gidx = list(d.groupby("patient_id").indices.values()); nd = sum(days.values())
    k_conf = max(1, round(CONFIRM_MIN / (5 * eval_stride)))
    def confirm(fl):
        c = np.zeros_like(fl)
        for ix in gidx: c[ix[k_conf:]] = fl[ix[k_conf:]] & fl[ix[:-k_conf]]
        return c
    def evaluate(fl, ev):
        y = d[f"y_{ev}"].values; tp = (fl & (y == 1)).sum(); n_on, leads = lead_times(d, fl, onsets, ev); al, hit = 0, 0
        for ix in gidx:
            st = alert_starts(tsv[ix], fl[ix]); al += len(st); hit += int((y[ix][st] == 1).sum())
        return dict(precision=tp / max(fl.sum(), 1), recall=tp / max((pop & (y == 1)).sum(), 1), episode_precision=hit / max(al, 1), n_onsets=n_on,
                    onsets_caught_pct=100 * len(leads) / max(n_on, 1), median_lead_min=float(np.median(leads)) if leads else np.nan, alerts_per_patient_day=al / nd)
    rows, ops, thr_used, flags_bal = [], {}, {}, {}
    for name in ("Trend rule", "Wearable-only", "Fused"):
        for ev in EVENTS:
            sc, y = score[(name, ev)], d[f"y_{ev}"].values
            for op, (thr, _, _) in op_thresholds(y[pop], sc[pop]).items():
                if name == "Trend rule" and op != "balanced": continue
                fl = confirm(pop & (sc >= thr)); r_ = evaluate(fl, ev)
                ops.setdefault(f"{name}|{ev}", {})[op] = dict(threshold=thr, **{k: (None if pd.isna(v) else float(v)) for k, v in r_.items()})
                if op == "balanced":
                    thr_used[f"{name}|{ev}"] = thr; flags_bal[(name, ev)] = fl
                    rows.append(dict(event=ev, model=name, AUC=roc_auc_score(y[pop], sc[pop]), AP=average_precision_score(y[pop], sc[pop]), prevalence=y[pop].mean(), threshold=thr, **r_))
        fl = flags_bal[(name, "hyper")] | flags_bal[(name, "hypo")]; yany = (np.nan_to_num(d.y_hyper.values) + np.nan_to_num(d.y_hypo.values)) > 0
        tp = (fl & yany).sum(); row = dict(event="any", model=name, prevalence=yany[pop].mean(), precision=tp / max(fl.sum(), 1), recall=tp / max((yany & pop).sum(), 1))
        if name != "Trend rule":
            sc = np.fmax(score[(name, "hyper")], score[(name, "hypo")]); row.update(AUC=roc_auc_score(yany[pop], sc[pop]), AP=average_precision_score(yany[pop], sc[pop]))
        rows.append(row)
    res = pd.DataFrame(rows); res.insert(2, "operating_point", "balanced")
    pd.DataFrame([dict(event=k.split("|")[1], model=k.split("|")[0], operating_point=op, **v) for k, o in ops.items() for op, v in o.items()]).round(4).to_csv(out / "events_operating_points.csv", index=False)

    # ---- cold start: first 24 h of a patient's data, before personal-history features exist (does the EHR help most here?)
    cold = {}
    for name in ("Wearable-only", "Fused"):
        for ev in EVENTS:
            m_ = pop & (d.idx.values < 288); y_ = d[f"y_{ev}"].values
            cold[f"{name}|{ev}"] = dict(n=int(m_.sum()), prevalence=float(y_[m_].mean()), AUC=float(roc_auc_score(y_[m_], score[(name, ev)][m_])), AP=float(average_precision_score(y_[m_], score[(name, ev)][m_])))
    print("cold start:", json.dumps(cold, indent=0))

    # ---- glucose forecast bands at +60 and +120 min (fused features), out-of-fold
    fc, fstats = {}, {}
    for h, col in ((60, "target"), (120, "target120")):
        has_t = (d[col].notna() & d.glucose.notna()).values; q = {a: np.full(len(d), np.nan) for a in QUANTILES}; delta = (d[col] - d.glucose).values
        for k in range(folds):
            tr = (fold != k) & has_t & d.train.values; te = (fold == k) & has_t
            for a in QUANTILES:
                m = lgb.LGBMRegressor(**dict(REG, alpha=a)).fit(d.loc[tr, FUSED], delta[tr]); q[a][te] = d.glucose.values[te] + m.predict(d.loc[te, FUSED])
        fc[h] = np.sort(np.vstack([q[a] for a in QUANTILES]), axis=0); t = d[col].values
        e = np.maximum(fc[h][0][has_t] - t[has_t], t[has_t] - fc[h][2][has_t]); w = float(np.quantile(e, 0.8))   # conformalised quantile regression: widen to 80% coverage
        fc[h][0] -= w; fc[h][2] += w
        fstats[f"{h}min"] = dict(band="80% (conformalised)", widen_mg_dl=w, empirical_coverage=float(np.mean((t[has_t] >= fc[h][0][has_t]) & (t[has_t] <= fc[h][2][has_t]))),
                                  median_MAE=float(np.mean(np.abs(t[has_t] - fc[h][1][has_t]))), persistence_MAE=float(np.mean(np.abs(t[has_t] - d.glucose.values[has_t]))))
        print(f"[events] forecast +{h} min: {fstats[f'{h}min']}", flush=True)
    Q = fc[60]
    risk = pd.DataFrame(dict(patient_id=d.patient_id.values, timestamp=d.timestamp.values, glucose=d.glucose.values,
                             p_hyper=score[("Fused", "hyper")], p_hypo=score[("Fused", "hypo")],
                             pw_hyper=score[("Wearable-only", "hyper")], pw_hypo=score[("Wearable-only", "hypo")],
                             fc_lo=Q[0], fc_mid=Q[1], fc_hi=Q[2], fc2_lo=fc[120][0], fc2_mid=fc[120][1], fc2_hi=fc[120][2]))
    risk.to_parquet(out / "event_risk.parquet", index=False)
    res.round(4).to_csv(out / "events.csv", index=False)
    json.dump(dict(profile=profile, window_min=WINDOW * 5, n_patients=int(d.patient_id.nunique()), n_rows=int(pop.sum()), patient_days=round(nd, 1),
                   thresholds=thr_used, confirm_min=CONFIRM_MIN, confirm_rows=k_conf, patient_fold=patient_fold, features=FUSED,
                   forecast=fstats, ops=ops, cold_start=cold, sustain_min=SUSTAIN * 5,
                   n_onsets={ev: int((onsets.kind == ev).sum()) for ev in EVENTS},
                   note="Thresholds maximise F1 on pooled out-of-fold scores; lead time = earliest alert within 2 h before an onset."),
              open(out / "events.json", "w"), indent=1)
    print(res.round(3).to_string()); print(json.dumps(fstats, indent=1))
    return res

_BOOSTERS = {}
def booster(profile, event, fold):
    k = (profile, event, fold)
    if k not in _BOOSTERS: _BOOSTERS[k] = lgb.Booster(model_file=str(data_dir(profile) / "results" / "event_models" / f"{event}_fold{fold}.txt"))
    return _BOOSTERS[k]

HISTORY_ROWS = 900     # 3 days of 5-min bins: covers the longest look-back (72 h low-glucose count)
def score_stream(profile, history, fold, confirm_min=CONFIRM_MIN):
    """Real-time inference on a stream: `history` = fused-timeline rows received so far (nothing later). Returns dict per event with
    the risk now, the risk CONFIRM_MIN earlier, and whether glucose is currently in range."""
    X = patient_frame(history.tail(HISTORY_ROWS)); back = int(round(confirm_min / 5))
    rows = X.iloc[[max(len(X) - 1 - back, 0), len(X) - 1]]; inr = rows.in_range.values
    out = {"in_range": bool(inr[1])}
    for ev in EVENTS:
        pr = booster(profile, ev, fold).predict(rows[FUSED].astype(float)); out[ev] = (float(pr[1]) if inr[1] else np.nan, float(pr[0]) if inr[0] else np.nan)
    return out

def predict_row(profile, row, event, fold, overrides=None):
    """Risk for one feature row, optionally with feature overrides (what-if analysis; association learned by the model, not causal)."""
    r = row[FUSED].astype(float).copy()
    for k, v in (overrides or {}).items(): r[k] = v
    return float(booster(profile, event, fold).predict(r)[0])

def explain(profile, patient_frame_row, event, fold, top=6):
    """SHAP-style contributions (log-odds) of the fused model of `fold` for one feature row. Returns [(feature, value, contrib)]."""
    c = booster(profile, event, fold).predict(patient_frame_row[FUSED].astype(float), pred_contrib=True)[0][:-1]
    order = np.argsort(-np.abs(c))[:top]
    return [(FUSED[i], float(patient_frame_row[FUSED[i]].iloc[0]), float(c[i])) for i in order]

if __name__ == "__main__":
    prof = sys.argv[1] if len(sys.argv) > 1 else "sample"
    run(prof, eval_stride=1 if prof == "sample" else 2, train_stride=3 if prof == "sample" else 6)
