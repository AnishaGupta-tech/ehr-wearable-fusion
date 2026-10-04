"""Patient Twin dashboard (Streamlit).  Run:  streamlit run dashboard/app.py
All data is synthetic. Genotypes, polygenic score and devices are SIMULATED."""
import sys, json, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.io as pio, streamlit as st
from plotly.subplots import make_subplots
from config import ROOT, ANCHOR

st.set_page_config(page_title="Patient Twin", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
STATUS = {"t2d": "Type 2 diabetes", "prediabetes": "Prediabetes", "none": "No diabetes"}

# ------------------------------------------------------------------ sidebar + theme
profiles = sorted([p.name for p in (ROOT / "data").iterdir() if (p / "fused_timeline.parquet").exists()], reverse=True)
if not profiles: st.error("No data yet. Run: python run_pipeline.py --profile sample"); st.stop()
with st.sidebar:
    st.markdown('<div class="brand"><span class="logo">PT</span><div><b>Patient Twin</b><small>EHR × wearable fusion</small></div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Cohort overview", "Patient view", "Model results"], label_visibility="collapsed")
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
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.17, 0.17, 0.12], vertical_spacing=0.035,
                            subplot_titles=("Glucose (mg/dL)", "Heart rate (bpm)", "Steps per 5 min", "Sleep"))
        fig.add_hrect(y0=70, y1=180, fillcolor=GREEN, opacity=0.09, line_width=0, row=1, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.glucose, line=dict(color=BLUE, width=1.6), name="Glucose", hovertemplate="%{y:.0f} mg/dL<extra></extra>"), row=1, col=1)
        meals = w[w.carbs_logged > 0]
        fig.add_trace(go.Scatter(x=meals.timestamp, y=[min(g.min() if len(g) else 70, 70) - 12] * len(meals), mode="markers", name="Logged meal",
                                 marker=dict(symbol="triangle-up", color=ORANGE, size=9), text=meals.carbs_logged.round().astype(int).astype(str) + " g carbs", hoverinfo="text+x"), row=1, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.hr, line=dict(color=RED, width=1), name="Heart rate", hovertemplate="%{y:.0f} bpm<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Bar(x=w.timestamp, y=w.steps, marker_color=GREEN, name="Steps"), row=3, col=1)
        for code, name, col in [(2, "Light", "#9bbfe6"), (3, "Deep", BLUE), (4, "REM", PURPLE), (1, "Awake", ORANGE)]:
            s = w[w.sleep_stage == code]; fig.add_trace(go.Bar(x=s.timestamp, y=[1] * len(s), marker_color=col, name=name, width=300000), row=4, col=1)
        style(fig, 700); fig.update_layout(barmode="stack", bargap=0, legend=dict(orientation="h", y=-0.03, x=0, font=dict(size=11)))
        fig.update_yaxes(visible=False, row=4, col=1); fig.update_annotations(x=0, xanchor="left")
        st.plotly_chart(fig, width="stretch")
        st.markdown('<p class="note">Shaded band = target range 70–180 mg/dL. Orange triangles = meals the patient logged; some meals go unlogged, as in real life.</p>', unsafe_allow_html=True)
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
    st.markdown('<p class="note">Caveat: the EHR drives the simulator, so these gains show the pipeline can recover the link, not that it holds in real patients.</p>', unsafe_allow_html=True)
