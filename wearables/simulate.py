"""Generate RAW device streams for one patient, conditioned on EHR-derived parameters.

Glucose comes from the simglucose UVA/Padova ODE (T1DPatient used as a generic glucose-insulin model, with
a PI "endogenous secretion" controller standing in for beta-cell function). Behaviour streams (sleep, steps,
HR, ...) are generated first; poor sleep and low prior-day activity lower next-day insulin sensitivity.
Output: long format (patient_id, source_device, metric, start_time, end_time, value, unit) at native device rates.
"""
import numpy as np, pandas as pd
from types import SimpleNamespace
from scipy.integrate import ode
from simglucose.patient.t1dpatient import T1DPatient, Action
from config import ANCHOR

CGM, WATCH, LOG = "Dexcom G7 (simulated)", "Apple Watch (simulated)", "Manual log / Insulin pen (simulated)"
STAGE = {"InBed": 0, "Awake": 1, "Core": 2, "Deep": 3, "REM": 4}     # numeric codes in `value` for metric sleep_stage
COLS = ["patient_id", "source_device", "metric", "start_time", "end_time", "value", "unit"]
DT = pd.Timedelta
KDOSE = 1.0

def _behaviour(p, days, rng):
    """Per-day schedule: sleep, steps minute series, meals. Day d covers [t0+d, t0+d+1)."""
    n = (days + 1) * 1440                              # +1 warm-up day
    steps = np.zeros(n); sleep_mask = np.zeros(n, bool); segs = []; daily = []
    prev_steps = p["steps_mean"]
    for d in range(days + 1):
        hours = float(np.clip(rng.normal(p["sleep_need"], 0.9) - 0.4 * (prev_steps < 4000), 4.0, 9.5))
        eff = float(np.clip(rng.normal(0.89, 0.05) - 0.02 * max(0, 6.5 - hours), 0.65, 0.98))
        total = int(hours * 60 / eff)
        s0 = d * 1440 + int(rng.normal(-60, 55))      # bedtime ~23:00 the evening before day d's wake-up
        s0 = max(s0, d * 1440 - 240); s1 = s0 + total
        a, b = max(s0, 0), min(s1, n); sleep_mask[a:b] = True
        # stage segments: ~90 min cycles, deep early, REM late, awakenings scaled by (1-eff)
        t = s0; segs.append((s0, s1, "InBed")); cyc = 0
        while t < s1:
            L = int(rng.normal(90, 10)); end = min(t + L, s1); frac = (t - s0) / max(total, 1)
            pieces = [("Core", 0.45 - 0.1 * frac), ("Deep", max(0.28 - 0.25 * frac, 0.03)), ("REM", 0.15 + 0.3 * frac)]
            tt = t; order = [pieces[0], pieces[1], pieces[0], pieces[2]]
            for name, w in order:
                dur = max(int(L * w * 0.9 * rng.uniform(0.7, 1.3)), 3); e = min(tt + dur, end)
                if e > tt: segs.append((tt, e, name))
                tt = e
                if rng.random() < (1 - eff) * 0.9 and tt < end:
                    e = min(tt + int(rng.uniform(2, 12)), end); segs.append((tt, e, "Awake")); tt = e
            if tt < end: segs.append((tt, end, "Core"))
            t = end
        # steps: walking bouts in waking hours
        target = float(np.clip(p["steps_mean"] * np.exp(rng.normal(0, 0.28)), 800, 22000)); acc = 0
        bouts = int(rng.integers(5, 12))
        for _ in range(bouts):
            start = d * 1440 + int(rng.choice([rng.normal(8 * 60, 40), rng.normal(12.5 * 60, 50), rng.normal(17.5 * 60, 70), rng.normal(14 * 60, 120), rng.normal(20 * 60, 60)]))
            dur = int(rng.integers(3, 35)); cad = rng.uniform(60, 125) if rng.random() < 0.85 else rng.uniform(125, 165)
            for m in range(start, start + dur):
                if 0 <= m < n and not sleep_mask[m]: steps[m] += max(0, rng.normal(cad, 12))
        cur = steps[d * 1440:(d + 1) * 1440].sum()
        if cur > 0: steps[d * 1440:(d + 1) * 1440] *= target / cur
        prev_steps = steps[d * 1440:(d + 1) * 1440].sum()
        daily.append(dict(day=d, sleep_hours=hours, efficiency=eff, steps=prev_steps, bed=s0, wake=s1))
    steps = np.round(steps)
    return steps, sleep_mask, segs, pd.DataFrame(daily)

def _meals(p, days, rng):
    out = []
    for d in range(days + 1):
        base = d * 1440
        plan = [(rng.normal(7.5, 0.6), 0.27), (rng.normal(12.7, 0.7), 0.33), (rng.normal(19.0, 0.9), 0.32)]
        if rng.random() < 0.45: plan.append((rng.normal(15.5, 1.0), 0.10))
        if rng.random() < 0.2: plan.append((rng.normal(21.5, 0.7), 0.10))
        for h, share in plan:
            cho = float(np.clip(p["cho_daily"] * share * rng.lognormal(0, 0.25), 8, 150))
            out.append((base + int(h * 60), cho))
    return out

def simulate_patient(pid, p, days, seed, tmpl_row, calib=False):
    rng = np.random.default_rng(seed)
    t0 = ANCHOR + DT(days=1)                            # window starts the day AFTER the last possible lab date (no leakage)
    warm = DT(days=1); n = (days + 1) * 1440
    steps, asleep, sleep_segs, daily = _behaviour(p, days, rng)
    meals = _meals(p, days, rng)
    # ---- daily sensitivity nudge: poor sleep + low prior-day activity -> lower next-day sensitivity
    day_sens = np.ones(days + 1)
    for d in range(1, days + 1):
        pr = daily.iloc[d - 1]; cur = daily.iloc[d]
        sleep_pen = 0.07 * np.clip((p["sleep_need"] - cur.sleep_hours) / 1.5, 0, 1.5) + 0.05 * np.clip((0.88 - cur.efficiency) / 0.1, 0, 1.5)
        act_pen = 0.08 * np.clip((p["steps_mean"] - pr.steps) / max(p["steps_mean"], 1), 0, 1)
        day_sens[d] = 1 - sleep_pen - act_pen
    # ---- patient ODE
    prm = tmpl_row.copy(); BW = prm.BW; u_ss = prm.u2ss * BW / 6000
    base = dict(Vmx=prm.Vmx, kp3=prm.kp3, k1=prm.k1, m4=prm.m4, kp1=prm.kp1, Vm0=prm.Vm0)
    prm["ke1"], prm["ke2"] = p["ke1"], p["ke2"]; prm["kp1"] = base["kp1"] * p["kp1_scale"]
    prm["k1"] = base["k1"] * p["k1_scale"]; prm["m4"] = base["m4"] * p["m4_scale"]
    prm["Vmx"] = base["Vmx"] * p["sens"]; prm["kp3"] = base["kp3"] * p["sens"]
    pt = T1DPatient(prm, t0=0)
    pt._params = SimpleNamespace(**prm.to_dict())          # attribute access without per-step pandas overhead
    box = [0.0]                                            # endogenous (portal/plasma) insulin secretion, pmol/kg/min
    def model(t, x, action, params, lq, lf):
        d = T1DPatient.model(t, x, action, params, lq, lf); d[5] += box[0]; return d
    pt._odesolver = ode(model).set_integrator("dopri5"); pt._odesolver.set_initial_value(pt.init_state, 0)
    gref = p["gref"]; Vg = prm.Vg
    cap = 1.0 + 7.0 * p["beta"] * (1.0 if not p["sulfonylurea"] else 1.3)
    kp, ki = 0.3 + 1.0 * p["beta"], 0.00015 + 0.0003 * p["beta"]
    floor = 0.6 * u_ss if p["sulfonylurea"] else 0.0       # SU keeps secreting at low glucose -> hypo risk
    exo_basal = 0.0
    if p["on_insulin"]: exo_basal = u_ss * 0.35
    I = 0.0; act = 0.0
    insulin_events = []
    G = np.zeros(n); u_hist = np.zeros(n); pending = []; bol = np.zeros(n + 60); endo = np.zeros(n + 60)
    cho_rate = np.zeros(n + 60)
    for m, cho in meals:                                  # eat over 12 min
        for k in range(12): cho_rate[min(m + k, n - 1)] += cho / 12
    for m, cho in meals:                                  # endogenous meal-time insulin release (beta-cell response)
        dose = KDOSE * p["beta"] ** 1.3 * cho / float(tmpl_row["CR"]) / max(p["sens"], 0.3) ** 0.5
        for k in range(30): endo[min(max(m + k - 5, 0), n - 1)] += dose / 30
        if p["on_insulin"] and rng.random() < 0.88:
            dose = cho / 11.0 * rng.uniform(0.7, 1.15); t_in = m - int(rng.integers(0, 15))
            pending.append((max(t_in, 0), dose)); insulin_events.append((max(t_in, 0), dose, "bolus"))
    for t_in, dose in pending:
        for k in range(10): bol[t_in + k] += dose / 10
    for t in range(n):
        d = t // 1440
        act = act * np.exp(-1 / 90) + steps[t] / 100.0 * (1 - np.exp(-1 / 90))
        s_t = day_sens[d] * (1 + 1.2 * np.clip(act, 0, 1.0))
        pt._params.Vmx = base["Vmx"] * p["sens"] * s_t; pt._params.kp3 = base["kp3"] * p["sens"] * s_t
        gp = pt.state[3] / Vg
        e = (gp - gref) / gref
        I = float(np.clip(I + e, -150, 1500 if p["beta"] > 0.3 else 700))
        u = u_ss * (1.0 + kp * e + ki * I) if gp > 80 else u_ss * max(0.2, 1 + kp * e)
        u = float(np.clip(u, max(floor, 0.1 * u_ss), cap * u_ss)) if gp > 70 or floor == 0 else float(max(floor, 0))
        u_endo = u + endo[t] * float(np.clip((gp - 70) / 40, 0, 1))
        box[0] = u_endo * 6000 / BW
        pt.step(Action(CHO=cho_rate[t], insulin=exo_basal + bol[t]))
        G[t] = pt.observation.Gsub; u_hist[t] = u_endo
    if calib: return float(G[1440:].mean())
    # ---- build raw streams -------------------------------------------------------------
    stamp = lambda m: t0 - warm + DT(minutes=int(m))
    rows = []
    def add(dev, metric, st, en, val, unit):
        rows.append((pid, dev, metric, st, en, val, unit))
    day0 = 1440                                           # drop warm-up day from outputs
    # CGM: 5-min readings, AR(1) noise, warm-up gap, dropouts
    cgm_t = np.arange(day0, n, 5); noise = np.zeros(len(cgm_t)); 
    for i in range(1, len(cgm_t)): noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 4.0)
    cgm = np.clip(G[cgm_t] * (1 + rng.normal(0, 0.015)) + noise, 40, 400)
    keep = np.ones(len(cgm_t), bool); keep[:6] = False                  # 30-min warm-up on first insertion
    if days >= 10: keep[(10 * 288):(10 * 288 + 6)] = False             # sensor change day 10
    for _ in range(rng.poisson(0.5 * days)):                           # dropouts 15-120 min
        s = int(rng.integers(0, len(cgm_t))); keep[s:s + int(rng.integers(3, 25))] = False
    for i in np.where(keep)[0]:
        ts = stamp(cgm_t[i]); add(CGM, "glucose", ts, ts, round(float(cgm[i]), 1), "mg/dL")
    # heart rate: 1 min when active, 5-10 min irregular at rest
    rhr_day = p["rhr_base"] + np.array([+3.0 * (daily.sleep_hours[d] < 6) + 0.0004 * (p["steps_mean"] - daily.steps[d]) for d in range(days + 1)])
    hr_true = np.zeros(n)
    for t in range(n):
        d = t // 1440; h = (t % 1440) / 60
        circ = 4 * np.sin(2 * np.pi * (h - 9) / 24)
        sleepdip = -9 if asleep[t] else 0
        post_meal = 0
        hr_true[t] = rhr_day[d] + 5 + circ + sleepdip + 0.38 * steps[t] + 8 * (G[t] > 180) * 0.3
    act_flag = pd.Series(steps).rolling(10, min_periods=1).max().values > 0
    t = day0
    while t < n:
        if act_flag[t]: ts = stamp(t); add(WATCH, "heart_rate", ts, ts, round(float(hr_true[t] + rng.normal(0, 2.5)), 0), "count/min"); t += 1
        else:
            ts = stamp(t); add(WATCH, "heart_rate", ts, ts, round(float(hr_true[t] + rng.normal(0, 2.5)), 0), "count/min")
            t += int(rng.integers(5, 11))
    # steps: aggregate consecutive active minutes into irregular segments (HealthKit style); active energy per active minute
    m = day0
    while m < n:
        if steps[m] > 0:
            e = m
            while e + 1 < n and steps[e + 1] > 0 and e - m < 25: e += 1
            add(WATCH, "steps", stamp(m), stamp(e + 1), float(steps[m:e + 1].sum()), "count"); m = e + 1
        else: m += 1
    for m in range(day0, n):
        if steps[m] > 0:
            kcal = 0.045 * steps[m] * (BW / 70) * rng.uniform(0.85, 1.15) + max(hr_true[m] - rhr_day[m // 1440] - 5, 0) * 0.012
            add(WATCH, "active_energy", stamp(m), stamp(m + 1), round(float(kcal), 3), "kcal")
    # daily/night metrics
    for d in range(1, days + 1):
        dr = daily.iloc[d]; base_t = d * 1440
        bed, wake = int(dr.bed), int(dr.wake)
        add(WATCH, "resting_heart_rate", stamp(base_t + 600), stamp(base_t + 600), round(float(rhr_day[d] + rng.normal(0, 1.2)), 0), "count/min")
        for _ in range(rng.integers(3, 6)):
            tt = base_t + int(rng.uniform(8 * 60, 21 * 60))
            add(WATCH, "hrv_rmssd", stamp(tt), stamp(tt), round(float(max(p["hrv_base"] * rng.lognormal(-0.1, 0.25) * (0.85 if steps[tt] > 0 else 1), 5)), 1), "ms")
        for tt in np.sort(rng.integers(bed, wake, size=int(rng.integers(4, 9)))):
            tt = int(np.clip(tt, 0, n - 1))
            add(WATCH, "hrv_rmssd", stamp(tt), stamp(tt), round(float(max(p["hrv_base"] * 1.25 * (dr.efficiency / 0.9) * rng.lognormal(0, 0.2), 5)), 1), "ms")
        for tt in range(bed + 20, wake, int(rng.integers(10, 16))):
            add(WATCH, "spo2", stamp(tt), stamp(tt), round(float(np.clip(p["spo2_base"] + rng.normal(0, 0.9) - (0.6 if dr.efficiency < 0.8 else 0), 85, 100)), 0), "%")
        add(WATCH, "wrist_temp_deviation", stamp(wake - 5), stamp(wake - 5),
            round(float(rng.normal(0, 0.22) + 0.08 * (G[max(0, bed - 600):bed].mean() - 140) / 40), 2), "degC")
        for (s0, s1, nm) in [s for s in sleep_segs if bed <= s[0] <= wake]:
            if s1 > 0 and nm and s0 < n:
                add(WATCH, "sleep_stage", stamp(max(s0, 0)), stamp(min(s1, n)), float(STAGE[nm]), "stage_code")
    # manual logs: under-logging (meals missed with prob 1-meal_log_p, carbs mis-estimated); insulin pen
    for mt, cho in meals:
        if mt < day0: continue
        if rng.random() < p["meal_log_p"]:
            add(LOG, "carbs_logged", stamp(mt + int(rng.integers(0, 20))), stamp(mt + int(rng.integers(0, 20))), round(float(cho * rng.lognormal(0, 0.25)), 0), "g")
    for tt, dose, kind in insulin_events:
        if tt >= day0 and rng.random() < p["insulin_log_p"]:
            add(LOG, "insulin_bolus", stamp(tt), stamp(tt), round(float(dose), 1), "U")
    if p["on_insulin"]:
        for d in range(1, days + 1):
            if rng.random() < 0.9:
                add(LOG, "insulin_basal", stamp(d * 1440 + 22 * 60), stamp(d * 1440 + 22 * 60), round(float(exo_basal * 1440), 0), "U")
    df = pd.DataFrame(rows, columns=COLS)
    df["start_time"] = pd.to_datetime(df.start_time); df["end_time"] = pd.to_datetime(df.end_time)
    truth = pd.DataFrame({"timestamp": [stamp(i) for i in range(day0, n)], "glucose_true": G[day0:]})
    return df.sort_values(["start_time", "metric"]).reset_index(drop=True), truth
