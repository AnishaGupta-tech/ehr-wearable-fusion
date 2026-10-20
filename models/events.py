"""Adverse-event model: will glucose leave 70-180 mg/dL within the next 2 hours?

Population: timestamps where glucose is currently in range (we predict NEW events, not ones already happening).
Labels:     hyper = max(glucose over t+5..t+120 min) > 180 ;  hypo = min(...) < 70.
Models:     LightGBM classifiers, patient-grouped CV.  Wearable-only vs fused (wearables + full EHR) vs a trend rule.
Also fits 10/50/90% quantile regressors for glucose at t+60 min (forecast band) and keeps the per-fold fused
models so the dashboard can explain an alert (SHAP contributions).
"""
import json, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
from config import data_dir
import features as F
from models.ablation import WEARABLE_FEATS, MODELS

HIGH, LOW = 180.0, 70.0
WINDOW = 24            # 24 x 5 min = 2 h
LEAD_MAX_MIN = 120
QUIET = 12             # onset = out of range after 1 h fully in range
EXTRA_WEARABLE = ["glucose_min60", "glucose_max60", "glucose_min180", "glucose_max180", "glucose_roc5", "glucose_roc60", "glucose_accel",
                  "min_since_logged_bolus", "steps_120"]
WEARABLE = WEARABLE_FEATS + EXTRA_WEARABLE
FUSED = MODELS["Fully fused"] + EXTRA_WEARABLE
FEATURE_SETS = {"Wearable-only": WEARABLE, "Fused": FUSED}
CONFIRM_MIN = 10       # an alert needs risk above threshold now AND 10 min earlier (suppresses single-sample blips)
EVENTS = ("hyper", "hypo")
CLF = dict(objective="binary", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, verbose=-1, n_jobs=8)
REG = dict(CLF, objective="quantile", n_estimators=250)
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
    fut = g.iloc[::-1]
    fmax = fut.rolling(WINDOW, min_periods=WINDOW // 2).max().iloc[::-1].shift(-1)
    fmin = fut.rolling(WINDOW, min_periods=WINDOW // 2).min().iloc[::-1].shift(-1)
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
    hy, ho = quiet & (g > HIGH), quiet & (g < LOW)
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
                   forecast=fstats, ops=ops,
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

HISTORY_ROWS = 220     # > 12 h of 5-min bins: enough for every rolling feature (longest look-back is the 720-min meal/bolus clip)
def score_stream(profile, history, fold):
    """Real-time inference on a stream: `history` = fused-timeline rows received so far (nothing later). Returns dict per event with
    the risk now, the risk CONFIRM_MIN earlier, and whether glucose is currently in range."""
    X = patient_frame(history.tail(HISTORY_ROWS)); back = CONFIRM_MIN // 5
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
