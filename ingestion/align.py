"""Ingestion & alignment: RAW device streams (long, native rates) -> shared 5-min timeline -> fused with EHR.

    Raw Device Streams -> load_raw_streams -> align_to_timeline -> fuse_with_ehr -> Fused Patient Timeline
"""
import numpy as np, pandas as pd
from pathlib import Path
from config import data_dir
import features as F

SLEEP_NONE = -1
MAX_GLUCOSE_GAP_BINS = 3          # interpolate gaps <= 15 min only

def load_raw_streams(patient_id, profile="sample", raw_dir=None):
    raw_dir = Path(raw_dir) if raw_dir else data_dir(profile) / "raw_wearables"
    df = pd.read_parquet(raw_dir / f"{patient_id}.parquet")
    df["start_time"] = pd.to_datetime(df.start_time); df["end_time"] = pd.to_datetime(df.end_time)
    return df

def _fill_small_gaps(s, max_bins):
    """Linear interpolation only inside NaN runs no longer than max_bins. Returns filled series and imputed mask."""
    isn = s.isna(); run_id = (isn != isn.shift()).cumsum(); run_len = isn.groupby(run_id).transform("sum")
    inter = s.interpolate(limit_area="inside")
    ok = isn & (run_len <= max_bins) & inter.notna()
    out = s.copy(); out[ok] = inter[ok]; return out, ok

def _nights(raw):
    """One row per night from InBed spans: availability time (wake), hours asleep, efficiency."""
    sl = raw[raw.metric == "sleep_stage"]; inbed = sl[sl.value == 0].sort_values("start_time")
    rows = []
    for _, r in inbed.iterrows():
        seg = sl[(sl.value.isin([2, 3, 4])) & (sl.start_time >= r.start_time) & (sl.end_time <= r.end_time)]
        asleep = (seg.end_time - seg.start_time).dt.total_seconds().sum() / 3600
        tib = (r.end_time - r.start_time).total_seconds() / 3600
        rows.append(dict(start=r.start_time, avail=r.end_time, hours=asleep, eff=asleep / tib if tib else np.nan))
    return pd.DataFrame(rows)

def _asof_series(grid, times, values, name):
    """Forward-fill an irregular series onto grid (value usable only from its timestamp on)."""
    if len(times) == 0: return pd.Series(np.nan, index=grid, name=name)
    s = pd.Series(np.asarray(values, float), index=pd.DatetimeIndex(times)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(grid)).ffill().reindex(grid).rename(name)

def align_to_timeline(raw, freq="5min"):
    pid = raw.patient_id.iloc[0]; fr = pd.Timedelta(freq)
    g = raw[raw.metric == "glucose"]
    start = g.start_time.min().floor("D"); end = g.start_time.max().floor(freq) + fr
    grid = pd.date_range(start, end, freq=freq, inclusive="left")
    tl = pd.DataFrame(index=grid); tl.index.name = "timestamp"
    # glucose: nearest reading within half a bin, small gaps interpolated, flags
    gs = g.set_index("start_time").value.sort_index()
    near = pd.merge_asof(pd.DataFrame({"t": grid}), gs.reset_index().rename(columns={"start_time": "t", "value": "glucose"}),
                         on="t", direction="nearest", tolerance=fr / 2).set_index("t").glucose
    filled, imp = _fill_small_gaps(near, MAX_GLUCOSE_GAP_BINS)
    tl["glucose"] = filled.values; tl["glucose_imputed"] = imp.values.astype(int)
    last_real = pd.Series(grid.where(near.notna().values), index=grid).ffill()
    tl["device_gap_minutes"] = ((pd.Series(grid, index=grid) - last_real).dt.total_seconds() / 60).fillna(0).clip(upper=24 * 60).values
    # heart rate: mean per bin; ffill <=2 bins, raw-missing flag
    hr = raw[raw.metric == "heart_rate"].set_index("start_time").value.resample(freq).mean().reindex(grid)
    tl["hr_missing"] = hr.isna().astype(int).values; tl["hr"] = hr.ffill(limit=2).values
    # steps (segments spread uniformly over their minutes) and active energy: sums
    st = raw[raw.metric == "steps"]; per_min = []
    for s, e, v in zip(st.start_time, st.end_time, st.value):
        idx = pd.date_range(s, e, freq="1min", inclusive="left")
        if len(idx): per_min.append(pd.Series(v / len(idx), index=idx))
    steps = pd.concat(per_min).groupby(level=0).sum().resample(freq).sum().reindex(grid).fillna(0) if per_min else pd.Series(0.0, index=grid)
    tl["steps"] = steps.values
    ae = raw[raw.metric == "active_energy"].set_index("start_time").value.resample(freq).sum().reindex(grid).fillna(0)
    tl["active_energy"] = ae.values
    # sleep stage covering the interval start: paint InBed first, finer stages over it
    sl = raw[raw.metric == "sleep_stage"].sort_values("value"); minute = pd.date_range(start, end, freq="1min", inclusive="left")
    code = pd.Series(SLEEP_NONE, index=minute, dtype=float)
    for s, e, v in zip(sl.start_time, sl.end_time, sl.value):
        code.loc[max(s, start):min(e, end) - pd.Timedelta(minutes=1)] = v
    tl["sleep_stage"] = code.reindex(grid).fillna(SLEEP_NONE).astype(int).values
    # manual logs
    for metric, col in [("carbs_logged", "carbs_logged"), ("insulin_bolus", "insulin_bolus_logged"), ("insulin_basal", "insulin_basal_logged")]:
        x = raw[raw.metric == metric].set_index("start_time").value
        tl[col] = x.resample(freq).sum().reindex(grid).fillna(0).values if len(x) else 0.0
    # daily / nightly metrics, forward-filled from the moment they would be known
    rhr = raw[raw.metric == "resting_heart_rate"].sort_values("start_time")
    tl["resting_hr"] = _asof_series(grid, rhr.start_time, rhr.value, "resting_hr").values
    trend = []
    for t in rhr.start_time: 
        h = rhr[rhr.start_time <= t].tail(7)
        trend.append(np.polyfit(np.arange(len(h)), h.value.values, 1)[0] if len(h) >= 3 else 0.0)
    tl["rhr_trend_7d"] = _asof_series(grid, rhr.start_time, trend, "t").values
    hv = raw[raw.metric == "hrv_rmssd"].sort_values("start_time")
    tl["hrv_rmssd_latest"] = _asof_series(grid, hv.start_time, hv.value, "h").values
    wt = raw[raw.metric == "wrist_temp_deviation"]
    tl["wrist_temp_dev"] = _asof_series(grid, wt.start_time, wt.value, "w").values
    nights = _nights(raw)
    if len(nights):
        tl["sleep_hours_last_night"] = _asof_series(grid, nights.avail, nights.hours, "a").values
        tl["sleep_efficiency_last_night"] = _asof_series(grid, nights.avail, nights.eff, "b").values
        sp = raw[raw.metric == "spo2"]; mins = []
        for _, n in nights.iterrows():
            v = sp[(sp.start_time >= n.start) & (sp.start_time <= n.avail)].value
            mins.append(v.min() if len(v) else np.nan)
        tl["spo2_night_min"] = _asof_series(grid, nights.avail, mins, "c").values
    else:
        for c in ("sleep_hours_last_night", "sleep_efficiency_last_night", "spo2_night_min"): tl[c] = np.nan
    tl.insert(0, "patient_id", pid)
    return tl.reset_index()

def fuse_with_ehr(timeline, ehr_tables):
    """Attach EHR features valid at each timestamp. Labs: only results dated strictly before the day of the timestamp.
    Conditions / medications: only those with onset / start date before that day."""
    pid = timeline.patient_id.iloc[0]
    pat = ehr_tables["patients"].set_index("patient_id", drop=False).loc[pid]
    labs, conds, meds, gen = (ehr_tables[k][ehr_tables[k].patient_id == pid] for k in ("labs", "conditions", "medications", "genomics"))
    days = pd.to_datetime(timeline.timestamp).dt.normalize(); out = {}
    for d in days.unique():
        c = conds[pd.to_datetime(conds.onset_date) < d]; m = meds[pd.to_datetime(meds.start_date) < d]
        p = pat.copy()
        if pd.notna(pat.diabetes_onset_date): p["years_since_dx"] = max((d - pd.Timestamp(pat.diabetes_onset_date)).days / 365.25, 0)
        out[d] = {**F.snapshot_features(pat), **F.ehr_features(p, labs, c, m, gen, asof=d)}
    feat = pd.DataFrame.from_dict(out, orient="index")
    return pd.concat([timeline.reset_index(drop=True), feat.reindex(days.values).reset_index(drop=True)], axis=1)

def build_timeseries(profile="sample", freq="5min"):
    """Stage-1 driver: raw streams -> timeseries.parquet (derived) and fused_timeline.parquet."""
    d = data_dir(profile); tables = {k: pd.read_csv(d / "ehr" / f"{k}.csv") for k in ("patients", "labs", "conditions", "medications", "genomics")}
    tls, fused = [], []
    for pid in tables["patients"].patient_id:
        tl = align_to_timeline(load_raw_streams(pid, profile), freq); tls.append(tl); fused.append(fuse_with_ehr(tl, tables))
    pd.concat(tls).to_parquet(d / "timeseries.parquet", index=False)
    pd.concat(fused).to_parquet(d / "fused_timeline.parquet", index=False)
    print(f"[ingestion] {profile}: {sum(map(len, tls))} timeline rows -> timeseries.parquet, fused_timeline.parquet")

if __name__ == "__main__":
    import sys; build_timeseries(sys.argv[1] if len(sys.argv) > 1 else "sample")
