"""Patient Twin dashboard (Streamlit).  Run:  streamlit run dashboard/app.py
All data is synthetic. Genotypes, polygenic score and devices are SIMULATED."""
import sys, json, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio, streamlit as st
from plotly.subplots import make_subplots
from config import ROOT, ANCHOR
from models import events as EV
import features as FT

st.set_page_config(page_title="Patient Twin", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
STATUS = {"t2d": "Type 2 diabetes", "prediabetes": "Prediabetes", "none": "No diabetes"}

# ------------------------------------------------------------------ sidebar + theme
profiles = sorted([p.name for p in (ROOT / "data").iterdir() if (p / "fused_timeline.parquet").exists()], reverse=True)
if not profiles: st.error("No data yet. Run: python run_pipeline.py --profile sample"); st.stop()
with st.sidebar:
    st.markdown('<div class="brand"><span class="logo">PT</span><div><b>Patient Twin</b><small>EHR × wearable fusion</small></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Cohort overview", "Triage board", "Patient view", "Live monitor", "Model results"], label_visibility="collapsed")
    st.markdown('<div class="side-label">Dataset</div>', unsafe_allow_html=True)
    profile = st.selectbox("Dataset", profiles, label_visibility="collapsed", help="'sample' = 50 patients (quick); 'full' = 800 patients.")
    st.markdown('<div class="side-label">Accessibility</div>', unsafe_allow_html=True)
    theme = st.selectbox("Theme", ["Light", "Dark", "High contrast"], label_visibility="collapsed")
    big = st.toggle("Larger text"); cb = st.toggle("Colour-blind-friendly colours", help="Okabe-Ito palette.")
    st.markdown('<div class="side-note">All data is synthetic (Synthea + simulation). Genotypes, polygenic score and device streams are <b>simulated</b>.</div>', unsafe_allow_html=True)

PAL = dict(blue="#2b6cb0", orange="#d9822b", green="#2f9e6e", red="#c9453b", grey="#8792a2", purple="#8265b8")
if cb: PAL = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", grey="#999999", purple="#CC79A7")
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = (PAL[k] for k in ("blue", "orange", "green", "red", "grey", "purple"))
TH = {"Light": dict(bg="#f6f7f9", card="#ffffff", text="#16202c", muted="#667385", line="#e4e8ee", soft="#eef2f7", accent=BLUE, grid="#edf0f4"),
      "Dark": dict(bg="#0d141c", card="#151e29", text="#e7edf4", muted="#97a5b6", line="#243142", soft="#1b2634", accent="#6aa7e8", grid="#243142"),
      "High contrast": dict(bg="#000000", card="#0a0a0a", text="#ffffff", muted="#ffe066", line="#ffffff", soft="#141414", accent="#66b3ff", grid="#444444")}[theme]
pio.templates.default = "plotly_dark" if theme != "Light" else "plotly_white"
t = TH
_N = ['[data-testid="stSidebar"] .element-container:has(.brand)+.element-container [role="radiogroup"]', '[data-testid="stSidebar"] .stElementContainer:has(.brand)+.stElementContainer [role="radiogroup"]']
def _nav(rule): return ",".join(f"{n} {rule.split('|')[0]}" for n in _N) + "{" + rule.split('|')[1] + "}"
NAVCSS = "\n".join([_nav("label|padding:.5rem .7rem;border-radius:8px;margin-bottom:2px;width:100%"), _nav("label:hover|background:var(--soft)"),
    _nav('label[data-baseweb="radio"]>div:first-child|display:none'), _nav("label:has(input:checked)|background:var(--soft);box-shadow:inset 3px 0 0 var(--accent);font-weight:600")])
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:{t['bg']};--card:{t['card']};--text:{t['text']};--muted:{t['muted']};--line:{t['line']};--soft:{t['soft']};--accent:{t['accent']}}}
html{{font-size:{'118%' if big else '100%'}}}
html,body,.stApp,[class*="css"],button,input,select,textarea{{font-family:'Inter',-apple-system,'Segoe UI',sans-serif!important}}
.stApp{{background:var(--bg);color:var(--text)}}
header[data-testid="stHeader"]{{background:transparent}} [data-testid="stToolbarActions"],[data-testid="stAppDeployButton"],.stAppDeployButton,.stDeployButton,[data-testid="stMainMenu"],[data-testid="stDecoration"],#MainMenu,footer{{display:none!important}}
[data-testid="stExpandSidebarButton"]{{display:flex!important;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:4px;margin:6px}} [data-testid="stExpandSidebarButton"] *{{color:var(--text)!important}}
.block-container{{padding:2.2rem 2.6rem 4rem;max-width:1360px}}
.stApp p,.stApp label,.stApp li,.stApp span,.stApp div[data-testid="stMarkdownContainer"]{{color:var(--text)}}
[data-testid="stSidebar"]{{background:var(--card);border-right:1px solid var(--line)}}
[data-testid="stSidebar"] .block-container{{padding:1.4rem 1.1rem}}
.brand{{display:flex;gap:.7rem;align-items:center;margin-bottom:1.4rem}}
.brand .logo{{width:36px;height:36px;border-radius:9px;background:var(--accent);color:#fff!important;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.85rem}}
.brand b{{display:block;font-size:1.02rem;color:var(--text)}} .brand small{{color:var(--muted);font-size:.74rem}}
.side-label{{font-size:.68rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:600;margin:1.3rem 0 .4rem}}
.side-note{{font-size:.76rem;color:var(--muted);line-height:1.5;margin-top:1.6rem;padding-top:1rem;border-top:1px solid var(--line)}} .side-note b{{color:var(--text)}}
{NAVCSS}
[data-testid="stSidebar"] label[data-baseweb="radio"]>div:first-child{{display:none!important}}
[data-testid="stSidebar"] label[data-baseweb="radio"]{{padding:.5rem .7rem;border-radius:8px;width:100%;margin-bottom:2px}}
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){{background:var(--soft);box-shadow:inset 3px 0 0 var(--accent);font-weight:600}}
[data-baseweb="select"]>div,[data-baseweb="select"]>div>div,[data-baseweb="select"] input{{background:var(--card)!important;color:var(--text)!important}}
[data-baseweb="select"] svg{{fill:var(--muted)}}
label[data-baseweb="checkbox"]>div:first-child{{background:var(--muted)!important;opacity:.55}}
label[data-baseweb="checkbox"]:has(input:checked)>div:first-child{{background:var(--accent)!important;opacity:1}}
.page-head h1{{font-size:1.65rem;font-weight:700;letter-spacing:-.02em;margin:0;color:var(--text)}}
.page-head p{{color:var(--muted);margin:.15rem 0 1.2rem;font-size:.95rem}} .page-head{{margin-bottom:.2rem}} .page-head+div{{margin-top:-.6rem}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:.4rem 0 1.4rem}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.kpi .l{{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}}
.kpi .v{{font-size:1.6rem;font-weight:650;letter-spacing:-.02em;margin-top:4px;color:var(--text);line-height:1.15}}
.kpi .v small{{font-size:.8rem;color:var(--muted);font-weight:500;margin-left:3px}}
.kpi .s{{font-size:.76rem;color:var(--muted);margin-top:3px}} .kpi.good .s{{color:#2f9e6e}} .kpi.bad .s{{color:#c9453b}}
.tags{{display:flex;flex-wrap:wrap;gap:6px;margin:.2rem 0 1rem}}
.tag{{font-size:.76rem;padding:3px 10px;border-radius:999px;background:var(--soft);color:var(--text);border:1px solid var(--line);font-weight:500}}
.tag.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}} .tag.warn{{border-color:#c9453b;color:#c9453b}}
.section{{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-weight:600;margin:1.6rem 0 .5rem}}
.callout{{background:var(--soft);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:8px;padding:.7rem .95rem;font-size:.88rem;color:var(--text)}}
.callout.sim{{border-left-color:#d9822b}}
.note{{font-size:.8rem;color:var(--muted);margin:.4rem 0 0}}
[data-baseweb="tab-list"]{{gap:4px;border-bottom:1px solid var(--line)}}
[data-baseweb="tab"]{{padding:.55rem .95rem!important;border-radius:8px 8px 0 0;font-weight:500;background:transparent!important}}
[data-baseweb="tab"][aria-selected="true"]{{color:var(--accent)!important;font-weight:600}} [data-baseweb="tab-highlight"]{{background:var(--accent)!important;height:2px}}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{{background:var(--card)!important;border:1px solid var(--line)!important;border-radius:8px!important}}
[data-testid="stDataFrame"]{{border:1px solid var(--line);border-radius:10px;overflow:hidden}}
[data-testid="stExpander"]{{border:1px solid var(--line)!important;border-radius:10px!important;background:var(--card)}}
[data-testid="stPlotlyChart"]{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 8px}}
:focus-visible{{outline:3px solid #d9822b!important;outline-offset:2px}}
</style>""", unsafe_allow_html=True)
D = ROOT / "data" / profile

def head(title, sub): st.markdown(f'<div class="page-head"><h1>{html.escape(title)}</h1><p>{html.escape(sub)}</p></div>', unsafe_allow_html=True)
def section(x): st.markdown(f'<div class="section">{html.escape(x)}</div>', unsafe_allow_html=True)
def kpis(items):
    h = '<div class="kpis">'
    for it in items:
        label, value = it[0], it[1]; unit = it[2] if len(it) > 2 else ""; sub = it[3] if len(it) > 3 else ""; cls = it[4] if len(it) > 4 else ""
        h += f'<div class="kpi {cls}"><div class="l">{html.escape(str(label))}</div><div class="v">{html.escape(str(value))}<small>{html.escape(unit)}</small></div>' + (f'<div class="s">{html.escape(sub)}</div>' if sub else "") + '</div>'
    st.markdown(h + '</div>', unsafe_allow_html=True)
def style(f, h=320, title=None):
    f.update_layout(height=h, margin=dict(l=12, r=12, t=44 if title else 16, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", size=12, color=t["text"]), title=dict(text=title, x=0.01, font=dict(size=13, color=t["muted"])) if title else None,
                    hoverlabel=dict(font_family="Inter"))
    f.update_xaxes(gridcolor=t["grid"], linecolor=t["line"], zeroline=False); f.update_yaxes(gridcolor=t["grid"], linecolor=t["line"], zeroline=False)
    for a in f.layout.annotations: a.font = dict(size=12, color=t["muted"]); a.x = a.x if a.xanchor != "center" else a.x
    return f

@st.cache_data
def load_ehr(profile): return {k: pd.read_csv(D / "ehr" / f"{k}.csv") for k in ("patients", "labs", "conditions", "medications", "genomics")}
@st.cache_data
def load_timeline(profile, pid): return pd.read_parquet(D / "timeseries.parquet", filters=[("patient_id", "==", pid)])
@st.cache_data
def load_raw(profile, pid): return pd.read_parquet(D / "raw_wearables" / f"{pid}.parquet")
@st.cache_data
def load_risk(profile, pid):
    f = D / "results" / "event_risk.parquet"
    return pd.read_parquet(f, filters=[("patient_id", "==", pid)]) if f.exists() else None
@st.cache_data
def load_event_meta(profile):
    f = D / "results" / "events.json"; return json.load(open(f)) if f.exists() else None
@st.cache_data
def load_fused(profile, pid): return pd.read_parquet(D / "fused_timeline.parquet", filters=[("patient_id", "==", pid)]).sort_values("timestamp").reset_index(drop=True)
@st.cache_data
def load_row(profile, pid, ts): X = EV.patient_frame(load_fused(profile, pid)); return X[X.timestamp == ts].reset_index(drop=True)
@st.cache_data
def load_explanation(profile, pid, ts, event, fold): return EV.explain(profile, load_row(profile, pid, ts), event, fold)
@st.cache_data
def load_all_risk(profile, op):
    """All patients' out-of-fold risk with confirmed-alert flags for the chosen operating point."""
    r = pd.read_parquet(D / "results" / "event_risk.parquet"); m = load_event_meta(profile); k = m["confirm_rows"]
    for ev in ("hyper", "hypo"):
        th = m["ops"][f"Fused|{ev}"][op]["threshold"]; hi = r[f"p_{ev}"] >= th
        r[f"alert_{ev}"] = hi & hi.groupby(r.patient_id).shift(k, fill_value=False)
        r[f"ratio_{ev}"] = r[f"p_{ev}"] / th
    return r

OPLAB = {"sensitive": "Sensitive: catch most events, more alerts", "balanced": "Balanced", "precise": "Precise: fewer false alarms"}
_em = load_event_meta(profile)
if _em:
    with st.sidebar:
        st.markdown('<div class="side-label">Alert sensitivity</div>', unsafe_allow_html=True)
        op = st.selectbox("Alert sensitivity", list(OPLAB), index=1, format_func=OPLAB.get, label_visibility="collapsed", help="Trade-off between missed events and false alarms. Each setting's measured performance is on the Model results page.")
def thr_of(emeta, ev): return emeta["ops"][f"Fused|{ev}"][op]["threshold"]

# plain-language names for model inputs: feature -> (label, value format)
FEAT_TEXT = {"glucose": ("Current glucose", "{:.0f} mg/dL"), "glucose_roc15": ("Glucose change, last 15 min", "{:+.0f} mg/dL"),
    "glucose_roc30": ("Glucose change, last 30 min", "{:+.0f} mg/dL"), "glucose_lag15": ("Glucose 15 min ago", "{:.0f} mg/dL"), "glucose_lag30": ("Glucose 30 min ago", "{:.0f} mg/dL"),
    "glucose_lag60": ("Glucose 1 h ago", "{:.0f} mg/dL"), "glucose_mean60": ("Average glucose, last hour", "{:.0f} mg/dL"), "glucose_std60": ("Glucose variability, last hour", "{:.0f} mg/dL"),
    "min_since_logged_meal": ("Minutes since last logged meal", "{:.0f} min"), "carbs_logged_3h": ("Carbs logged, last 3 h", "{:.0f} g"), "bolus_logged_3h": ("Insulin bolus logged, last 3 h", "{:.1f} U"),
    "hr": ("Heart rate", "{:.0f} bpm"), "hr_mean30": ("Heart rate, last 30 min", "{:.0f} bpm"), "steps_30": ("Steps, last 30 min", "{:.0f}"), "steps_60": ("Steps, last hour", "{:.0f}"),
    "active_energy_60": ("Active energy, last hour", "{:.0f} kcal"), "tod_sin": ("Time of day", "{:.2f}"), "tod_cos": ("Time of day", "{:.2f}"), "asleep": ("Asleep", "{:.0f}"),
    "sleep_hours_last_night": ("Sleep last night", "{:.1f} h"), "sleep_efficiency_last_night": ("Sleep efficiency", "{:.2f}"), "hrv_rmssd_latest": ("HRV", "{:.0f} ms"),
    "resting_hr": ("Resting heart rate", "{:.0f} bpm"), "rhr_trend_7d": ("Resting HR trend, 7 d", "{:+.1f} bpm"), "spo2_night_min": ("Night SpO₂ minimum", "{:.0f} %"),
    "wrist_temp_dev": ("Wrist temperature deviation", "{:+.2f} °C"), "hba1c": ("HbA1c", "{:.1f} %"), "hba1c_latest": ("Latest HbA1c", "{:.1f} %"), "fasting_glucose": ("Fasting glucose", "{:.0f} mg/dL"),
    "bmi": ("BMI", "{:.1f}"), "age": ("Age", "{:.0f}"), "egfr": ("eGFR", "{:.0f}"), "homa_ir": ("HOMA-IR", "{:.1f}"), "on_insulin": ("On insulin", "{:.0f}"),
    "on_sulfonylurea": ("On sulfonylurea", "{:.0f}"), "on_metformin": ("On metformin", "{:.0f}"), "on_sglt2": ("On SGLT2 inhibitor", "{:.0f}"), "years_since_dx": ("Years since diagnosis", "{:.1f}"),
    "genetic_risk_score": ("Genetic risk score (sim.)", "{:.2f}"), "hba1c_slope_2y": ("HbA1c trend per year", "{:+.2f}"), "n_glucose_lowering_classes": ("Glucose-lowering drug classes", "{:.0f}")}
RECORD_FEATS = set(FT.SNAPSHOT_FEATS + FT.FULL_EXTRA)
def feat_text(f, v):
    if f.startswith("tod_"): return "Time of day", ""
    lab, fm = FEAT_TEXT.get(f, (f.replace("_", " ").capitalize(), "{:.2f}")); return lab, fm.format(v) if pd.notna(v) else "–"
ehr = load_ehr(profile); pat = ehr["patients"].copy(); pat["status_label"] = pat.diabetes_status.map(STATUS)

# ------------------------------------------------------------------ cohort
if page == "Cohort overview":
    head("Cohort overview", f"{len(pat)} synthetic patients · dataset: {profile}")
    kpis([("Patients", len(pat)), ("Type 2 diabetes", int((pat.diabetes_status == 't2d').sum())), ("Prediabetes", int((pat.diabetes_status == "prediabetes").sum())),
          ("Median HbA1c", f"{pat.hba1c.median():.1f}", "%"), ("Median BMI", f"{pat.bmi.median():.1f}")])
    a, b, c = st.columns(3)
    a.plotly_chart(style(go.Figure(go.Histogram(x=pat.hba1c, marker_color=BLUE, nbinsx=30)), 300, "HbA1c (%)"), width="stretch")
    b.plotly_chart(style(go.Figure(go.Histogram(x=pat.bmi, marker_color=GREEN, nbinsx=30)), 300, "BMI"), width="stretch")
    c.plotly_chart(style(go.Figure(go.Histogram(x=pat.genetic_risk_score, marker_color=ORANGE, nbinsx=25)), 300, "Polygenic risk score (simulated, 0–1)"), width="stretch")
    section("Patients")
    flt = st.multiselect("Diabetes status", list(STATUS.values()), default=list(STATUS.values()), label_visibility="collapsed")
    show = pat[pat.status_label.isin(flt)][["patient_id", "status_label", "age", "sex", "bmi", "hba1c", "egfr", "homa_ir", "genetic_risk_score"]]
    st.dataframe(show.rename(columns={"patient_id": "Patient", "status_label": "Status", "age": "Age", "sex": "Sex", "bmi": "BMI", "hba1c": "HbA1c", "egfr": "eGFR", "homa_ir": "HOMA-IR", "genetic_risk_score": "Genetic risk (sim.)"}),
                 hide_index=True, width="stretch", height=380)
    st.markdown('<p class="note">Open a patient from <b>Patient view</b> in the sidebar.</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ patient
elif page == "Patient view":
    head("Patient view", "Wearable signals, health record and genetics for one synthetic patient")
    f1, f2 = st.columns([1, 3])
    status = f1.selectbox("Diabetes status", ["All"] + list(STATUS.values()))
    sub = pat if status == "All" else pat[pat.status_label == status]
    label = sub.assign(l=sub.patient_id.str[:8] + "  ·  " + sub.age.astype(str) + sub.sex + "  ·  HbA1c " + sub.hba1c.astype(str)).set_index("l").patient_id
    pid = label[f2.selectbox("Patient", label.index)]; p = pat.set_index("patient_id").loc[pid]
    tl = load_timeline(profile, pid); raw = load_raw(profile, pid)
    cond = ehr["conditions"][ehr["conditions"].patient_id == pid]; meds = ehr["medications"][ehr["medications"].patient_id == pid]
    other = [c for c in cond[cond.status == "active"].condition if c not in ("Type 2 diabetes", "Prediabetes")]
    tags = f'<span class="tag primary">{STATUS[p.diabetes_status]}</span>' + "".join(f'<span class="tag {"warn" if c.startswith(("Diabetic", "Chronic", "Coronary")) else ""}">{html.escape(c)}</span>' for c in other)
    st.markdown(f'<div class="tags">{tags}</div>', unsafe_allow_html=True)
    kpis([("Age / sex", f"{p.age} {p.sex}"), ("HbA1c", p.hba1c, "%"), ("BMI", p.bmi), ("eGFR", p.egfr, "", "kidney function", "bad" if p.egfr < 60 else ""),
          ("HOMA-IR", p.homa_ir, "", "insulin resistance"), ("Genetic risk", p.genetic_risk_score, "", "simulated score, 0–1")])
    t_gl, t_rec, t_gen, t_dev = st.tabs(["Glucose & wearables", "Health record", "Genetics (simulated)", "Devices"])

    with t_gl:
        days = sorted(tl.timestamp.dt.date.unique()); c1, c2 = st.columns([3, 1.3])
        d0 = c1.select_slider("Start day", days, value=days[0]); nd = c2.radio("Window", [1, 2, 3, 7], index=1, horizontal=True, format_func=lambda x: f"{x}d")
        w = tl[(tl.timestamp >= pd.Timestamp(d0)) & (tl.timestamp < pd.Timestamp(d0) + pd.Timedelta(days=nd))]; g = w.glucose.dropna()
        tir = ((g >= 70) & (g <= 180)).mean() * 100
        kpis([("Mean glucose", f"{g.mean():.0f}", "mg/dL"), ("Time in range 70–180", f"{tir:.0f}", "%", "target ≥ 70%", "good" if tir >= 70 else "bad"),
              ("Time above 180", f"{(g > 180).mean()*100:.0f}", "%"), ("Time below 70", f"{(g < 70).mean()*100:.1f}", "%", "target < 4%", "good" if (g < 70).mean() < .04 else "bad")])
        risk, emeta = load_risk(profile, pid), load_event_meta(profile); has = risk is not None and emeta is not None
        if has:
            thH, thL = thr_of(emeta, "hyper"), thr_of(emeta, "hypo")
            rw = risk[(risk.timestamp >= w.timestamp.min()) & (risk.timestamp <= w.timestamp.max())].reset_index(drop=True)
            rw["ratio"] = np.fmax(rw.p_hyper / thH, rw.p_hypo / thL); kc = emeta["confirm_rows"]
            for e_, th_ in (("hyper", thH), ("hypo", thL)):
                hi_ = rw[f"p_{e_}"] >= th_; rw[f"alert_{e_}"] = hi_ & hi_.shift(kc, fill_value=False)
            section("Risk in the next 2 hours")
            at = st.select_slider("Assess risk at", options=list(rw.timestamp), value=rw.timestamp[int(rw.ratio.idxmax()) if rw.ratio.notna().any() else len(rw) - 1],
                                  format_func=lambda x: x.strftime("%a %d %b, %H:%M"), help="Starts at the moment of highest risk in this window.")
            r = rw[rw.timestamp == at].iloc[0]
            def card(label, pr, th, conf):
                if pd.isna(pr): return (label, "–", "", "glucose already out of range", "")
                st_ = "ALERT" if conf else ("above threshold, confirming" if pr >= th else "no alert")
                return (label, f"{pr*100:.0f}", "%", st_ + f" · alerts at ≥ {th*100:.0f}%", "bad" if conf else "good")
            fcs = lambda lo, mid, hi: ("–", "") if pd.isna(mid) else (f"{mid:.0f}", f"likely range {lo:.0f}–{hi:.0f}")
            kpis([card("Glucose > 180 within 2 h", r.p_hyper, thH, r.alert_hyper), card("Glucose < 70 within 2 h", r.p_hypo, thL, r.alert_hypo),
                  ("Glucose now", "–" if pd.isna(r.glucose) else f"{r.glucose:.0f}", "mg/dL"),
                  ("Forecast at +1 h", fcs(r.fc_lo, r.fc_mid, r.fc_hi)[0], "mg/dL", fcs(r.fc_lo, r.fc_mid, r.fc_hi)[1]),
                  ("Forecast at +2 h", fcs(r.fc2_lo, r.fc2_mid, r.fc2_hi)[0], "mg/dL", fcs(r.fc2_lo, r.fc2_mid, r.fc2_hi)[1])])
            fold = emeta["patient_fold"].get(pid)
            if fold is not None and r[["p_hyper", "p_hypo"]].notna().any():
                lead = "hyper" if (r.ratio == r.p_hyper / thH) else "hypo"
                names = {"hyper": "High glucose (> 180)", "hypo": "Low glucose (< 70)"}
                ev = st.radio("Why flagged", ["hyper", "hypo"], index=["hyper", "hypo"].index(lead), horizontal=True, format_func=names.get, key=f"why{pid}{at}")
                exp = load_explanation(profile, pid, at, ev, fold); rows_ = []
                for f_, v_, c_ in exp:
                    lab, val = feat_text(f_, v_); k_ = (f"{lab}: {val}" if val else lab) + ("  (health record)" if f_ in RECORD_FEATS else "")
                    prev = [i for i, x in enumerate(rows_) if x[0] == k_]
                    if prev: rows_[prev[0]] = (k_, rows_[prev[0]][1] + c_)
                    else: rows_.append((k_, c_))
                rows_.sort(key=lambda x: -abs(x[1]))
                top = [x for x in rows_ if x[1] > 0][:3]
                st.markdown('<div class="callout">' + (("<b>Main reasons for higher risk:</b> " + "; ".join(html.escape(x[0].split("  (")[0]) for x in top)) if top else "No input is pushing the risk up.") + '</div>', unsafe_allow_html=True)
                dfc = pd.DataFrame(rows_, columns=["f", "c"]).iloc[::-1]
                fc = go.Figure(go.Bar(x=dfc.c, y=dfc.f, orientation="h", marker_color=[RED if v > 0 else GREEN for v in dfc.c]))
                style(fc, 80 + 34 * len(dfc), "Contribution to risk (red raises, green lowers)"); fc.update_layout(xaxis_title="effect on risk (log-odds)"); st.plotly_chart(fc, width="stretch")
                with st.expander("What-if: change medication or activity (model-based, not medical advice)"):
                    row = load_row(profile, pid, at); cur = {k: int(row[k].iloc[0]) for k in ("on_metformin", "on_sglt2", "on_sulfonylurea", "on_dpp4", "on_insulin")}
                    wc = st.columns(5); ov = {}
                    for col_, (k, lab) in zip(wc, [("on_metformin", "Metformin"), ("on_sglt2", "SGLT2 inhibitor"), ("on_sulfonylurea", "Sulfonylurea"), ("on_dpp4", "DPP-4 inhibitor"), ("on_insulin", "Insulin")]):
                        ov[k] = int(col_.checkbox(lab, value=bool(cur[k]), key=f"wi{k}{pid}"))
                    ov["n_glucose_lowering_classes"] = sum(ov.values()); s60 = float(row.steps_60.iloc[0])
                    ov["steps_60"] = st.slider("Steps in the last hour", 0, 6000, int(min(s60, 6000)), 100, key=f"wis{pid}{at}"); ov["steps_30"] = ov["steps_60"] / 2
                    wi = pd.DataFrame([dict(Event=names[e_], Now=EV.predict_row(profile, row, e_, fold) * 100, **{"What-if": EV.predict_row(profile, row, e_, fold, ov) * 100}) for e_ in ("hyper", "hypo")])
                    wi["Change (points)"] = wi["What-if"] - wi.Now
                    st.dataframe(wi.round(1).rename(columns={"Now": "Risk now (%)", "What-if": "Risk with changes (%)"}), hide_index=True, width="stretch")
                    st.markdown('<p class="note">Re-scores the same moment with the edited inputs. The model learned associations from synthetic data; this is a hypothesis for a clinician to weigh, not a treatment recommendation.</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="note">Risk applies to moments when glucose is currently in range. Alert sensitivity: {OPLAB[op]}. An alert needs risk above threshold now and {emeta["confirm_min"]} min earlier. Decision support only; synthetic data.</p>', unsafe_allow_html=True)
            fcm = st.radio("Forecast band on chart", ["Off", "1 hour ahead", "2 hours ahead"], index=1, horizontal=True)
        R = dict(hr=3, st=4, sl=5) if has else dict(hr=2, st=3, sl=4)
        fig = make_subplots(rows=5 if has else 4, cols=1, shared_xaxes=True, row_heights=[0.4, 0.16, 0.16, 0.16, 0.12] if has else [0.5, 0.17, 0.17, 0.12], vertical_spacing=0.035,
                            subplot_titles=("Glucose (mg/dL)", "Event risk, next 2 h (%)", "Heart rate (bpm)", "Steps per 5 min", "Sleep") if has else ("Glucose (mg/dL)", "Heart rate (bpm)", "Steps per 5 min", "Sleep"))
        fig.add_hrect(y0=70, y1=180, fillcolor=GREEN, opacity=0.09, line_width=0, row=1, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.glucose, line=dict(color=BLUE, width=1.6), name="Glucose", hovertemplate="%{y:.0f} mg/dL<extra></extra>"), row=1, col=1)
        if has and fcm != "Off":
            hz, pre = (60, "fc") if fcm.startswith("1") else (120, "fc2"); fx = rw.timestamp + pd.Timedelta(minutes=hz)
            fig.add_trace(go.Scatter(x=fx, y=rw[pre + "_hi"], line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=fx, y=rw[pre + "_lo"], line=dict(width=0), fill="tonexty", fillcolor="rgba(130,101,184,.22)", name=f"Forecast range ({hz//60} h ahead)", hoverinfo="skip"), row=1, col=1)
            fig.add_trace(go.Scatter(x=fx, y=rw[pre + "_mid"], line=dict(color=PURPLE, width=1.1, dash="dot"), name=f"Forecast ({hz//60} h ahead)", hovertemplate="%{y:.0f} mg/dL<extra>forecast</extra>"), row=1, col=1)
        if has:
            fig.add_trace(go.Scatter(x=rw.timestamp, y=rw.p_hyper * 100, line=dict(color=ORANGE, width=1.4), name="Risk > 180", hovertemplate="%{y:.0f}%<extra>high</extra>"), row=2, col=1)
            fig.add_trace(go.Scatter(x=rw.timestamp, y=rw.p_hypo * 100, line=dict(color=RED, width=1.4), name="Risk < 70", hovertemplate="%{y:.0f}%<extra>low</extra>"), row=2, col=1)
            fig.add_hline(y=thH * 100, line=dict(color=ORANGE, width=1, dash="dash"), row=2, col=1); fig.add_hline(y=thL * 100, line=dict(color=RED, width=1, dash="dash"), row=2, col=1)
            for rr in range(1, 6): fig.add_vline(x=at.to_pydatetime(), line=dict(color=t["muted"], width=1, dash="dot"), row=rr, col=1)
        meals = w[w.carbs_logged > 0]
        fig.add_trace(go.Scatter(x=meals.timestamp, y=[min(g.min() if len(g) else 70, 70) - 12] * len(meals), mode="markers", name="Logged meal",
                                 marker=dict(symbol="triangle-up", color=ORANGE, size=9), text=meals.carbs_logged.round().astype(int).astype(str) + " g carbs", hoverinfo="text+x"), row=1, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.hr, line=dict(color=RED, width=1), name="Heart rate", hovertemplate="%{y:.0f} bpm<extra></extra>"), row=R["hr"], col=1)
        fig.add_trace(go.Bar(x=w.timestamp, y=w.steps, marker_color=GREEN, name="Steps"), row=R["st"], col=1)
        for code, name, col in [(2, "Light", "#9bbfe6"), (3, "Deep", BLUE), (4, "REM", PURPLE), (1, "Awake", ORANGE)]:
            s = w[w.sleep_stage == code]; fig.add_trace(go.Bar(x=s.timestamp, y=[1] * len(s), marker_color=col, name=name, width=300000), row=R["sl"], col=1)
        style(fig, 860 if has else 700); fig.update_layout(barmode="stack", bargap=0, legend=dict(orientation="h", y=-0.03, x=0, font=dict(size=11)))
        fig.update_yaxes(visible=False, row=R["sl"], col=1); fig.update_annotations(x=0, xanchor="left")
        st.plotly_chart(fig, width="stretch")
        st.markdown('<p class="note">Shaded band = target range 70–180 mg/dL. Orange triangles = meals the patient logged; some meals go unlogged, as in real life. Purple band = where the model expects glucose to be 1 or 2 hours later (80% range, calibrated on held-out patients).</p>', unsafe_allow_html=True)
        n = tl[tl.timestamp >= pd.Timestamp(d0)].iloc[0]; fmt = lambda v, r=1: "–" if pd.isna(v) else round(float(v), r)
        section("Overnight and daily signals")
        kpis([("Resting HR", fmt(n.resting_hr, 0), "bpm"), ("Sleep last night", fmt(n.sleep_hours_last_night), "h"), ("Night SpO₂ min", fmt(n.spo2_night_min, 0), "%"),
              ("Wrist temp deviation", fmt(n.wrist_temp_dev, 2), "°C"), ("HRV", fmt(n.hrv_rmssd_latest, 0), "ms")])

    with t_rec:
        r1, r2, r3 = st.tabs(["Conditions", "Lab trends", "Medications"])
        with r1:
            c = cond.copy(); c["onset_date"] = pd.to_datetime(c.onset_date)
            if len(c):
                c = c.sort_values("onset_date"); fig = go.Figure()
                for r in c.itertuples():
                    fig.add_trace(go.Scatter(x=[r.onset_date, ANCHOR], y=[r.condition] * 2, mode="lines+markers", showlegend=False, line=dict(color=BLUE if r.status == "active" else GREY, width=8),
                                             marker=dict(size=[11, 0]), hovertemplate=f"{r.condition}<br>onset {r.onset_date.date()} ({r.status})<extra></extra>"))
                style(fig, 90 + 42 * len(c)); fig.update_yaxes(autorange="reversed", showgrid=False); st.plotly_chart(fig, width="stretch")
                st.markdown('<p class="note">Blue = active, grey = resolved. Bars run from diagnosis to today.</p>', unsafe_allow_html=True)
                st.dataframe(c.assign(onset_date=c.onset_date.dt.date)[["condition", "onset_date", "status"]].rename(columns={"condition": "Condition", "onset_date": "Onset date", "status": "Status"}), hide_index=True, width="stretch")
            else: st.info("No recorded conditions.")
        with r2:
            l = ehr["labs"][ehr["labs"].patient_id == pid].copy(); l["date"] = pd.to_datetime(l.date)
            fig = make_subplots(rows=1, cols=3, subplot_titles=("HbA1c (%)", "LDL cholesterol (mg/dL)", "eGFR (mL/min/1.73m²)"))
            for i, (tt, col) in enumerate([("HbA1c", BLUE), ("LDL", ORANGE), ("eGFR", GREEN)], 1):
                s = l[l.test == tt].sort_values("date"); fig.add_trace(go.Scatter(x=s.date, y=s.value, mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=7), showlegend=False), row=1, col=i)
            style(fig, 330); fig.update_annotations(x=None); st.plotly_chart(fig, width="stretch")
            with st.expander("All lab results"): st.dataframe(l.sort_values(["test", "date"]).assign(date=lambda x: x.date.dt.date).drop(columns="patient_id"), hide_index=True, width="stretch")
        with r3:
            act = meds[meds.status == "active"].drop(columns="patient_id")
            st.dataframe(act.rename(columns=lambda c: c.replace("_", " ").title()), hide_index=True, width="stretch") if len(act) else st.info("No active medications.")
            if (meds.status != "active").any():
                section("Stopped"); st.dataframe(meds[meds.status != "active"].drop(columns="patient_id"), hide_index=True, width="stretch")

    with t_gen:
        st.markdown('<div class="callout sim"><b>Simulated data.</b> All genotypes and the polygenic score are generated for this demo. They are not real genetic data.</div>', unsafe_allow_html=True)
        gdf = ehr["genomics"][ehr["genomics"].patient_id == pid].drop(columns="patient_id").rename(columns={"snp_id": "SNP", "gene": "Gene", "genotype": "Genotype", "risk_allele_count": "Risk alleles (0–2)", "label": "Provenance"})
        section("Risk variants"); st.dataframe(gdf, hide_index=True, width="stretch")
        kpis([("Polygenic risk score", p.genetic_risk_score, "", "top quartile of cohort" if p.genetic_risk_flag else "below top quartile", "bad" if p.genetic_risk_flag else "")])

    with t_dev:
        rows = [{"Device": dev, "Status": "Connected", "Last sync": d.end_time.max().strftime("%Y-%m-%d %H:%M"), "Data types": ", ".join(sorted(d.metric.unique())), "Records": len(d)} for dev, d in raw.groupby("source_device")]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch"); st.markdown('<p class="note">Devices are simulated.</p>', unsafe_allow_html=True)
        section("Data quality")
        kpis([("Glucose coverage", f"{tl.glucose.notna().mean()*100:.1f}", "%"), ("Longest CGM gap", int(tl.device_gap_minutes.max()), "min"),
              ("Gaps filled (≤ 15 min)", f"{tl.glucose_imputed.mean()*100:.2f}", "%"), ("HR bins without reading", f"{tl.hr_missing.mean()*100:.0f}", "%")])

# ------------------------------------------------------------------ triage board
elif page == "Triage board":
    head("Triage board", "Which patients need attention right now? The twin ranks the whole panel by predicted risk of a glucose event in the next 2 hours")
    if _em is None or not (D / "results" / "event_risk.parquet").exists(): st.info("Run the events stage first: python run_pipeline.py --stages events"); st.stop()
    R_ = load_all_risk(profile, op); R_["any_alert"] = R_.alert_hyper | R_.alert_hypo
    per_t = R_.groupby("timestamp").any_alert.sum(); times = list(per_t.index)
    at = st.select_slider("Snapshot time (simulated clinic clock)", options=times, value=per_t.idxmax(), format_func=lambda x: x.strftime("%a %d %b, %H:%M"))
    snap = R_[R_.timestamp == at].merge(pat[["patient_id", "age", "sex", "diabetes_status", "hba1c"]], on="patient_id")
    snap["ratio"] = np.fmax(snap.ratio_hyper.fillna(0), snap.ratio_hypo.fillna(0)); snap = snap.sort_values("ratio", ascending=False)
    ops_ = _em["ops"]["Fused|hyper"][op], _em["ops"]["Fused|hypo"][op]; per100 = 100 * (ops_[0]["alerts_per_patient_day"] + ops_[1]["alerts_per_patient_day"])
    kpis([("Patients monitored", len(snap)), ("In alert now", int(snap.any_alert.sum()), "", f"{snap.any_alert.mean()*100:.0f}% of panel", "bad" if snap.any_alert.mean() > .2 else ""),
          ("High-glucose alerts", int(snap.alert_hyper.sum())), ("Low-glucose alerts", int(snap.alert_hypo.sum())),
          ("Expected workload", f"{per100:.0f}", "", "alerts per day per 100 patients")])
    def kind(r): return " + ".join(x for x, f in (("High", r.alert_hyper), ("Low", r.alert_hypo)) if f) or "–"
    tb = snap.head(25).assign(Alert=lambda x: x.apply(kind, axis=1), Patient=lambda x: x.patient_id.str[:8], Profile=lambda x: x.age.astype(str) + x.sex + " · " + x.diabetes_status.map(STATUS))
    st.dataframe(tb.assign(g=tb.glucose.round(0), h=(tb.p_hyper * 100).round(0), l=(tb.p_hypo * 100).round(0), f2=tb.fc2_mid.round(0))[["Patient", "Profile", "hba1c", "g", "h", "l", "f2", "Alert"]].rename(columns={
        "hba1c": "HbA1c", "g": "Glucose now", "h": "Risk > 180 (%)", "l": "Risk < 70 (%)", "f2": "Forecast +2 h"}), hide_index=True, width="stretch", height=640)
    st.markdown('<p class="note">Top 25 of the panel, highest risk first. Open a patient by the ID prefix in <b>Patient view</b>. Patients whose glucose is already out of range show no risk (the model scores new events only).</p>', unsafe_allow_html=True)

# ------------------------------------------------------------------ live monitor
elif page == "Live monitor":
    head("Live monitor", "Simulated real-time feed: the twin sees only data that has arrived, re-scores every tick, and looks 2 hours ahead")
    if _em is None: st.info("Run the events stage first: python run_pipeline.py --stages events"); st.stop()
    lab_ = pat.assign(l=pat.patient_id.str[:8] + "  ·  " + pat.age.astype(str) + pat.sex + "  ·  HbA1c " + pat.hba1c.astype(str)).set_index("l").patient_id
    pid = lab_[st.selectbox("Patient", lab_.index, key="live_patient")]; tlf = load_fused(profile, pid); fold = _em["patient_fold"][pid]
    risk_all = load_all_risk(profile, op); rk = risk_all[risk_all.patient_id == pid].reset_index(drop=True)
    if st.session_state.get("live_pid") != pid:
        st.session_state.update(live_pid=pid, live_i=min(300, len(tlf) - 1))
    c1, c2, c3, c4 = st.columns([1, 1.4, 1, 1.2]); playing = c1.toggle("Streaming", key="live_play")
    step = c2.select_slider("Speed", [1, 3, 6, 12], value=3, format_func=lambda x: f"{x*5} min/s")
    if c3.button("Restart"): st.session_state.live_i = 150
    if c4.button("Jump to next alert"):
        cur = tlf.timestamp[st.session_state.live_i]; nxt = rk[(rk.timestamp > cur + pd.Timedelta(hours=2)) & (rk.alert_hyper | rk.alert_hypo)]
        if len(nxt): st.session_state.live_i = max(int(tlf.index[tlf.timestamp == nxt.timestamp.iloc[0] - pd.Timedelta(minutes=45)][0]), 150)
    thH, thL = thr_of(_em, "hyper"), thr_of(_em, "hypo")

    @st.fragment(run_every="1s" if playing else None)
    def feed():
        if playing: st.session_state.live_i = min(st.session_state.live_i + step, len(tlf) - 1)
        i = st.session_state.live_i; now = tlf.timestamp[i]; hist = tlf.iloc[:i + 1]; sc = EV.score_stream(profile, hist, fold)
        last = hist.iloc[-1]; conf = {e: (not np.isnan(sc[e][0])) and sc[e][0] >= th and (not np.isnan(sc[e][1])) and sc[e][1] >= th for e, th in (("hyper", thH), ("hypo", thL))}
        st.markdown(f'<div class="callout"><b>Clock: {now.strftime("%a %d %b %Y, %H:%M")}</b> · {i + 1:,} readings received · nothing after this time is visible to the model</div>', unsafe_allow_html=True)
        def card(label, pr, th, c):
            if np.isnan(pr[0]): return (label, "–", "", "glucose already out of range", "")
            return (label, f"{pr[0]*100:.0f}", "%", ("ALERT" if c else "no alert") + f" · alerts at ≥ {th*100:.0f}%", "bad" if c else "good")
        rn = rk[rk.timestamp <= now].iloc[-1] if (rk.timestamp <= now).any() else None
        fcs = lambda p_: "–" if rn is None or pd.isna(rn[p_ + "_mid"]) else f"{rn[p_ + '_mid']:.0f}"
        kpis([card("Glucose > 180 within 2 h", sc["hyper"], thH, conf["hyper"]), card("Glucose < 70 within 2 h", sc["hypo"], thL, conf["hypo"]),
              ("Glucose now", "–" if pd.isna(last.glucose) else f"{last.glucose:.0f}", "mg/dL"), ("Forecast +1 h", fcs("fc"), "mg/dL"), ("Forecast +2 h", fcs("fc2"), "mg/dL")])
        win = hist[hist.timestamp >= now - pd.Timedelta(hours=8)]; fg = go.Figure()
        fg.add_hrect(y0=70, y1=180, fillcolor=GREEN, opacity=0.09, line_width=0)
        fg.add_trace(go.Scatter(x=win.timestamp, y=win.glucose, line=dict(color=BLUE, width=2), name="Glucose (received)"))
        if rn is not None and not pd.isna(rn.fc_mid):
            fx = [now + pd.Timedelta(minutes=60), now + pd.Timedelta(minutes=120)]
            fg.add_trace(go.Scatter(x=fx, y=[rn.fc_mid, rn.fc2_mid], mode="markers+lines", name="Forecast (80% range)", line=dict(color=PURPLE, dash="dot"), marker=dict(color=PURPLE, size=9),
                                    error_y=dict(type="data", symmetric=False, array=[rn.fc_hi - rn.fc_mid, rn.fc2_hi - rn.fc2_mid], arrayminus=[rn.fc_mid - rn.fc_lo, rn.fc2_mid - rn.fc2_lo], color=PURPLE)))
        al = rk[(rk.timestamp <= now) & (rk.timestamp >= now - pd.Timedelta(hours=8)) & (rk.alert_hyper | rk.alert_hypo)]
        fg.add_trace(go.Scatter(x=al.timestamp, y=al.glucose, mode="markers", name="Alert raised", marker=dict(symbol="triangle-up", color=RED, size=10, line=dict(color="#fff", width=1))))
        style(fg, 360); fg.update_layout(legend=dict(orientation="h", y=-0.12), yaxis_title="mg/dL", xaxis=dict(range=[now - pd.Timedelta(hours=8), now + pd.Timedelta(hours=2, minutes=20)]))
        fg.add_vline(x=now, line=dict(color=t["muted"], dash="dot")); st.plotly_chart(fg, width="stretch")
        section("Alert log (this patient, up to now)")
        ep = rk[(rk.timestamp <= now)].copy(); log = []
        for e_, nm, test in (("hyper", "Glucose > 180", lambda g_: g_ > 180), ("hypo", "Glucose < 70", lambda g_: g_ < 70)):
            a_ = ep[ep[f"alert_{e_}"]].timestamp; prev = None
            for tt in a_:
                if prev is not None and (tt - prev) <= pd.Timedelta(minutes=30): prev = tt; continue
                prev = tt; fut = tlf[(tlf.timestamp > tt) & (tlf.timestamp <= tt + pd.Timedelta(hours=2)) & (tlf.timestamp <= now)]
                full = tt + pd.Timedelta(hours=2) <= now; hit = fut[test(fut.glucose)]
                out_ = f"Confirmed: glucose {'rose above 180' if e_ == 'hyper' else 'fell below 70'} at {hit.timestamp.iloc[0].strftime('%H:%M')}" if len(hit) else ("False alarm" if full else "Pending (watching)")
                log.append({"Time": tt.strftime("%a %d %b %H:%M"), "Alert": nm, "Risk (%)": round(float(ep[ep.timestamp == tt][f"p_{e_}"].iloc[0]) * 100), "Outcome": out_})
        st.dataframe(pd.DataFrame(log[::-1]) if log else pd.DataFrame({"Alert log": ["No alerts yet"]}), hide_index=True, width="stretch", height=240)
        st.markdown('<p class="note">Each tick the latest readings are appended, features are rebuilt from history only and scored with the held-out-fold model. Press Streaming to play, or Jump to next alert.</p>', unsafe_allow_html=True)
    feed()

# ------------------------------------------------------------------ results
else:
    head("Model results", "Does adding health-record (EHR) data help predict glucose 60 minutes ahead?")
    rf = D / "results" / "ablation.csv"
    if not rf.exists(): st.info("Run the models stage first: python run_pipeline.py --stages models"); st.stop()
    res = pd.read_csv(rf, index_col=0); meta = json.load(open(D / "results" / "ablation.json"))
    base, snap, full = res.loc["Wearable-only", "MAE"], res.loc["Fused (snapshot EHR only)", "MAE"], res.loc["Fully fused", "MAE"]
    kpis([("Wearable only", f"{base:.2f}", "mg/dL", "average error"), ("+ EHR snapshot", f"{snap:.2f}", "mg/dL", f"{snap-base:+.2f} vs wearable only", "good"),
          ("+ full history & genetics", f"{full:.2f}", "mg/dL", f"{full-base:+.2f} vs wearable only", "good")])
    st.markdown(f'<div class="callout">Tested on <b>{meta["n_patients"]} unseen patients</b> (5-fold cross-validation grouped by patient, {meta["n_rows"]:,} predictions). Lower error is better.</div>', unsafe_allow_html=True)
    colors = {"Persistence": GREY, "Wearable-only": BLUE, "Fused (snapshot EHR only)": ORANGE, "Fully fused": GREEN}
    fig = go.Figure(go.Bar(x=res.index, y=res.MAE, marker_color=[colors[i] for i in res.index], text=res.MAE.round(2), textposition="outside", width=0.5))
    section("Average error by model"); style(fig, 340); fig.update_layout(yaxis=dict(range=[0, res.MAE.max() * 1.2], title="MAE (mg/dL)")); st.plotly_chart(fig, width="stretch")
    with st.expander("All metrics"): st.dataframe(res.round(3), width="stretch")
    with st.expander("Statistical tests (per-patient comparison)"):
        tt = pd.DataFrame(meta["paired_tests"]).T.rename(columns={"mean_MAE_gain": "Error reduction (mg/dL)", "pct_patients_improved": "% patients improved", "wilcoxon_p": "p-value"})
        st.dataframe(tt.round(4), width="stretch")
    with st.expander("Which inputs mattered most (fully fused model)"):
        s = pd.Series(meta["top_features"]["Fully fused"]).sort_values(); s = s / s.sum() * 100
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=BLUE)); style(fig, 440); fig.update_layout(xaxis_title="share of importance (%)"); st.plotly_chart(fig, width="stretch")
    ef = D / "results" / "events.csv"
    if ef.exists():
        ev = pd.read_csv(ef); em = json.load(open(D / "results" / "events.json")); section("Adverse-event alerts: glucose out of range within 2 hours")
        st.markdown(f'<div class="callout">Predicts whether glucose goes <b>above 180</b> or <b>below 70</b> in the next 2 hours, for moments when it is currently in range. {em["n_rows"]:,} test moments from {em["n_patients"]} unseen patients ({em["patient_days"]:.0f} patient-days).</div>', unsafe_allow_html=True)
        fu = ev[(ev.model == "Fused")].set_index("event"); wo = ev[(ev.model == "Wearable-only")].set_index("event")
        kpis([("High-glucose AUC", f"{fu.loc['hyper','AUC']:.2f}", "", f"{fu.loc['hyper','AUC']-wo.loc['hyper','AUC']:+.3f} vs wearable only"),
              ("Low-glucose AUC", f"{fu.loc['hypo','AUC']:.2f}", "", f"{fu.loc['hypo','AUC']-wo.loc['hypo','AUC']:+.3f} vs wearable only"),
              ("High events caught", f"{fu.loc['hyper','onsets_caught_pct']:.0f}", "%", f"median {fu.loc['hyper','median_lead_min']:.0f} min warning"),
              ("Low events caught", f"{fu.loc['hypo','onsets_caught_pct']:.0f}", "%", f"median {fu.loc['hypo','median_lead_min']:.0f} min warning"),
              ("Alerts per patient-day", f"{fu.loc['hyper','alerts_per_patient_day']+fu.loc['hypo','alerts_per_patient_day']:.1f}", "", "high + low")])
        shown = ev[ev.event != "any"].rename(columns={"event": "Event", "model": "Model", "AP": "Avg. precision", "precision": "Precision", "recall": "Recall", "episode_precision": "Alert-episode precision",
                                                      "onsets_caught_pct": "Events caught (%)", "median_lead_min": "Median warning (min)", "alerts_per_patient_day": "Alerts / patient-day"})
        st.dataframe(shown[["Event", "Model", "AUC", "Avg. precision", "Precision", "Recall", "Alert-episode precision", "Events caught (%)", "Median warning (min)", "Alerts / patient-day"]].round(3), hide_index=True, width="stretch")
        opf = D / "results" / "events_operating_points.csv"
        if opf.exists():
            section("Alert sensitivity: what each setting costs"); opd = pd.read_csv(opf); opd = opd[opd.model == "Fused"]
            st.dataframe(opd[["event", "operating_point", "recall", "episode_precision", "onsets_caught_pct", "median_lead_min", "alerts_per_patient_day"]].rename(columns={
                "event": "Event", "operating_point": "Setting", "recall": "Recall", "episode_precision": "Alert-episode precision", "onsets_caught_pct": "Events caught (%)",
                "median_lead_min": "Median warning (min)", "alerts_per_patient_day": "Alerts / patient-day"}).round(3), hide_index=True, width="stretch")
        mc = {"Trend rule": GREY, "Wearable-only": BLUE, "Fused": GREEN}; fig = go.Figure()
        for m_ in mc: d_ = ev[(ev.model == m_) & (ev.event != "any")]; fig.add_trace(go.Bar(x=d_.event.map({"hyper": "Above 180", "hypo": "Below 70"}), y=d_.AP, name=m_, marker_color=mc[m_], text=d_.AP.round(2), textposition="outside"))
        style(fig, 320, "Average precision (higher is better)"); fig.update_layout(barmode="group", yaxis=dict(range=[0, ev.AP.max() * 1.25])); st.plotly_chart(fig, width="stretch")
        if "cold_start" in em:
            cs = pd.DataFrame(em["cold_start"]).T.reset_index(); cs[["Model", "Event"]] = cs["index"].str.split("|", expand=True); cs["Event"] = cs.Event.map({"hyper": "Above 180", "hypo": "Below 70"})
            with st.expander("Day-one performance: first 24 hours of a patient's data (little wearable history yet)"):
                st.dataframe(cs[["Event", "Model", "AUC", "AP"]].rename(columns={"AP": "Avg. precision"}).round(3), hide_index=True, width="stretch")
                st.markdown('<p class="note">Once a patient has a day or more of CGM, their own recent history carries most of the signal and the health record adds little; in the first 24 h the record is a useful prior.</p>', unsafe_allow_html=True)
        f6, f12 = em["forecast"]["60min"], em["forecast"]["120min"]
        kpis([("Forecast +1 h error", f"{f6['median_MAE']:.1f}", "mg/dL", f"persistence {f6['persistence_MAE']:.1f}", "good"), ("Forecast +2 h error", f"{f12['median_MAE']:.1f}", "mg/dL", f"persistence {f12['persistence_MAE']:.1f}", "good"),
              ("Band coverage +1 h", f"{f6['empirical_coverage']*100:.0f}", "%", "target 80%"), ("Band coverage +2 h", f"{f12['empirical_coverage']*100:.0f}", "%", "target 80%")])
        st.markdown(f'<p class="note">Alert sensitivity setting: thresholds are chosen on pooled held-out scores (balanced = best F1). An alert needs risk above threshold now and {em["confirm_min"]} min earlier. A caught event means an alert fired in the 2 h before glucose first left range after an hour in range; alert-episode precision is the share of alert episodes followed by an event within 2 h. “Trend rule” = glucose + 2 × 30-min change. Forecast bands are conformalised on held-out predictions.</p>', unsafe_allow_html=True)
    st.markdown('<p class="note">Caveat: the EHR drives the simulator, so these gains show the pipeline can recover the link, not that it holds in real patients.</p>', unsafe_allow_html=True)
